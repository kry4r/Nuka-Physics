#pragma once
// ---------------------------------------------------------------------------
// Shared test meshes for the V-HACD convex-decomposition tests.
// ---------------------------------------------------------------------------
// Hand-authored closed triangle meshes (no asset files needed):
//   - UnitCubeMesh:   a solid axis-aligned cube spanning [-0.5, 0.5]^3 (convex).
//   - LShapeMesh:     an extruded L-shaped prism (clearly CONCAVE — a single
//                     convex hull would fill in the notch, so V-HACD must split
//                     it into multiple pieces). Stands in for a "robot finger"
//                     style concave collision mesh.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

namespace nuka::test {

struct TestMesh {
    std::vector<float>    vertices;  // x,y,z triples
    std::vector<uint32_t> indices;   // triangle indices
};

// Axis-aligned cube spanning [-0.5, 0.5]^3, 8 verts / 12 triangles, outward
// winding.
inline TestMesh UnitCubeMesh() {
    TestMesh m;
    m.vertices = {
        -0.5f, -0.5f, -0.5f,  // 0
         0.5f, -0.5f, -0.5f,  // 1
         0.5f,  0.5f, -0.5f,  // 2
        -0.5f,  0.5f, -0.5f,  // 3
        -0.5f, -0.5f,  0.5f,  // 4
         0.5f, -0.5f,  0.5f,  // 5
         0.5f,  0.5f,  0.5f,  // 6
        -0.5f,  0.5f,  0.5f,  // 7
    };
    m.indices = {
        0, 2, 1,  0, 3, 2,  // -Z
        4, 5, 6,  4, 6, 7,  // +Z
        0, 1, 5,  0, 5, 4,  // -Y
        3, 7, 6,  3, 6, 2,  // +Y
        0, 4, 7,  0, 7, 3,  // -X
        1, 2, 6,  1, 6, 5,  // +X
    };
    return m;
}

// Build a closed triangle mesh for an axis-aligned box [min, max] with outward
// winding, appended into the given mesh.
inline void AppendBox(TestMesh& m,
                      float minx, float miny, float minz,
                      float maxx, float maxy, float maxz) {
    const uint32_t base = static_cast<uint32_t>(m.vertices.size() / 3);
    const float v[8][3] = {
        {minx, miny, minz}, {maxx, miny, minz}, {maxx, maxy, minz}, {minx, maxy, minz},
        {minx, miny, maxz}, {maxx, miny, maxz}, {maxx, maxy, maxz}, {minx, maxy, maxz},
    };
    for (const auto& p : v) {
        m.vertices.push_back(p[0]);
        m.vertices.push_back(p[1]);
        m.vertices.push_back(p[2]);
    }
    const uint32_t f[12][3] = {
        {0, 2, 1}, {0, 3, 2},  {4, 5, 6}, {4, 6, 7},
        {0, 1, 5}, {0, 5, 4},  {3, 7, 6}, {3, 6, 2},
        {0, 4, 7}, {0, 7, 3},  {1, 2, 6}, {1, 6, 5},
    };
    for (const auto& tri : f) {
        m.indices.push_back(base + tri[0]);
        m.indices.push_back(base + tri[1]);
        m.indices.push_back(base + tri[2]);
    }
}

// Extruded L-shape: two overlapping boxes forming an L (concave) cross-section,
// extruded in Z. The re-entrant notch makes a single convex hull a poor fit, so
// V-HACD splits it into ~2 pieces. (The mesh is the union of two boxes; the
// internal shared faces are harmless to V-HACD's voxelization.)
inline TestMesh LShapeMesh() {
    TestMesh m;
    // Vertical bar of the L: x in [0,1], y in [0,3], z in [0,1]
    AppendBox(m, 0.0f, 0.0f, 0.0f, 1.0f, 3.0f, 1.0f);
    // Horizontal foot of the L: x in [0,3], y in [0,1], z in [0,1]
    AppendBox(m, 0.0f, 0.0f, 0.0f, 3.0f, 1.0f, 1.0f);
    return m;
}

} // namespace nuka::test
