// ---------------------------------------------------------------------------
// PBF particle-set -> marching-cubes isosurface mesher tests.
//
// runtime::fluid::MarchFluidSurface turns a fluid particle set into a watertight
// smooth render::MeshGeometry by marching the SPH density isosurface. These tests
// run the mesher as a PURE FUNCTION of positions (no physics step, no device) on a
// SYNTHETIC settled pool (a dense particle lattice filling an AABB) + a sphere of
// particles, and assert the mechanical bar:
//   1. DETERMINISM (D1): identical input -> byte-identical positions/indices/normals.
//   2. WATERTIGHT-ish: one connected component (not N blobs); edge-manifold (every
//      triangle edge shared by <=2 triangles); ~0 boundary (once-used) edges.
//   3. ENVELOPE: the mesh AABB surrounds the particles within ~h (no explode/collapse).
//   4. NORMALS: every normal finite + unit length, none zero/NaN.
//   5. SPHERE SANITY: a sphere of particles -> a roughly spherical closed shell.
// ---------------------------------------------------------------------------

#include "runtime/fluid/surface_mesher.hpp"

#include "import/cooker/fluid_cooker_types.hpp"  // ComputePbfDensities (rho0 calibration)
#include "math/vec3.hpp"
#include "render/render_world.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <utility>
#include <vector>

using nuka::math::Vec3;
using nuka::render::MeshGeometry;
using nuka::runtime::fluid::FluidSurfaceParams;
using nuka::runtime::fluid::MarchFluidSurface;

namespace {

// A settled pool: a cell-centered lattice of `nx*ny*nz` particles at `spacing`,
// the deterministic stand-in for a rested fluid block (the fluid cooker's layout).
std::vector<Vec3> SettledPool(int nx, int ny, int nz, float spacing, Vec3 corner) {
    std::vector<Vec3> pts;
    pts.reserve(static_cast<size_t>(nx) * ny * nz);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                pts.push_back({corner.x + (static_cast<float>(i) + 0.5f) * spacing,
                               corner.y + (static_cast<float>(j) + 0.5f) * spacing,
                               corner.z + (static_cast<float>(k) + 0.5f) * spacing});
    return pts;
}

// A solid ball of lattice particles within `radius` of `center`.
std::vector<Vec3> ParticleBall(Vec3 center, float radius, float spacing) {
    std::vector<Vec3> pts;
    const int r = static_cast<int>(std::ceil(radius / spacing)) + 1;
    for (int k = -r; k <= r; ++k)
        for (int j = -r; j <= r; ++j)
            for (int i = -r; i <= r; ++i) {
                const Vec3 q{center.x + static_cast<float>(i) * spacing,
                             center.y + static_cast<float>(j) * spacing,
                             center.z + static_cast<float>(k) * spacing};
                const Vec3 d = q - center;
                if (d.Dot(d) <= radius * radius) pts.push_back(q);
            }
    return pts;
}

// rho0 CALIBRATED numerically from the particle set via the shared host Poly6
// (ComputePbfDensities), the cooker's recipe: the deep-interior density. iso =
// 0.5*rho0 then sits in the steep part of the field where the gradient is strong.
FluidSurfaceParams PoolParams(const std::vector<Vec3>& pts, float spacing) {
    FluidSurfaceParams p;
    p.h = 2.0f * spacing;          // support spans ~2 rings.
    p.particle_mass = 1.0f;
    p.iso_fraction = 0.5f;
    p.cell_size = 0.5f * spacing;  // finer than h for a smooth surface.

    nuka::runtime::fluid::PbfParticleSet ps;
    ps.positions = pts;
    ps.particle_mass = p.particle_mass;
    nuka::runtime::fluid::PbfParams pp;
    pp.support_radius_h = p.h;
    const std::vector<float> rho = nuka::runtime::fluid::ComputePbfDensities(ps, pp);
    float rho_max = 0.0f;
    for (float r : rho) rho_max = std::max(rho_max, r);
    p.rest_density_rho0 = rho_max;
    return p;
}

struct MeshStats {
    uint32_t verts = 0, tris = 0;
    int components = 0;
    int boundary_edges = 0;   // edges used by exactly one triangle (a hole rim).
    int nonmanifold_edges = 0;// edges used by >2 triangles.
    Vec3 lo{}, hi{};
    float min_nlen = 1e30f, max_nlen = 0.0f;
    bool all_finite_unit = true;
};

// Union-find over vertices (welded by shared index) to count connected components.
int CountComponents(uint32_t nverts, const std::vector<uint32_t>& idx) {
    std::vector<uint32_t> parent(nverts);
    for (uint32_t i = 0; i < nverts; ++i) parent[i] = i;
    std::function<uint32_t(uint32_t)> find = [&](uint32_t a) {
        while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
        return a;
    };
    auto unite = [&](uint32_t a, uint32_t b) { parent[find(a)] = find(b); };
    for (size_t t = 0; t + 2 < idx.size(); t += 3) {
        unite(idx[t], idx[t + 1]);
        unite(idx[t + 1], idx[t + 2]);
    }
    // Count roots that are actually referenced by a triangle.
    std::vector<char> used(nverts, 0);
    for (uint32_t v : idx) used[v] = 1;
    int comps = 0;
    for (uint32_t i = 0; i < nverts; ++i)
        if (used[i] && find(i) == i) ++comps;
    return comps;
}

MeshStats Analyze(const MeshGeometry& m) {
    MeshStats s;
    s.verts = m.VertexCount();
    s.tris = m.TriangleCount();
    if (s.verts == 0) return s;
    s.lo = s.hi = Vec3{m.positions[0], m.positions[1], m.positions[2]};
    for (uint32_t v = 0; v < s.verts; ++v) {
        const Vec3 p{m.positions[v * 3], m.positions[v * 3 + 1], m.positions[v * 3 + 2]};
        s.lo.x = std::min(s.lo.x, p.x); s.lo.y = std::min(s.lo.y, p.y); s.lo.z = std::min(s.lo.z, p.z);
        s.hi.x = std::max(s.hi.x, p.x); s.hi.y = std::max(s.hi.y, p.y); s.hi.z = std::max(s.hi.z, p.z);
        const Vec3 nrm{m.normals[v * 3], m.normals[v * 3 + 1], m.normals[v * 3 + 2]};
        const float nl = std::sqrt(nrm.Dot(nrm));
        s.min_nlen = std::min(s.min_nlen, nl);
        s.max_nlen = std::max(s.max_nlen, nl);
        if (!std::isfinite(nl) || std::fabs(nl - 1.0f) > 1e-3f) s.all_finite_unit = false;
    }
    s.components = CountComponents(s.verts, m.indices);
    // Edge incidence: count triangles per undirected edge.
    std::map<std::pair<uint32_t, uint32_t>, int> edge_use;
    auto bump = [&](uint32_t a, uint32_t b) {
        if (a > b) std::swap(a, b);
        ++edge_use[{a, b}];
    };
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        bump(m.indices[t], m.indices[t + 1]);
        bump(m.indices[t + 1], m.indices[t + 2]);
        bump(m.indices[t + 2], m.indices[t]);
    }
    for (const auto& kv : edge_use) {
        if (kv.second == 1) ++s.boundary_edges;
        if (kv.second > 2) ++s.nonmanifold_edges;
    }
    return s;
}

void PrintStats(const char* label, const MeshStats& s) {
    std::printf(
        "[%s] verts=%u tris=%u components=%d boundary_edges=%d nonmanifold_edges=%d\n"
        "    bbox=[%.4f,%.4f,%.4f]..[%.4f,%.4f,%.4f]  |n| in [%.6f, %.6f]\n",
        label, s.verts, s.tris, s.components, s.boundary_edges, s.nonmanifold_edges,
        s.lo.x, s.lo.y, s.lo.z, s.hi.x, s.hi.y, s.hi.z, s.min_nlen, s.max_nlen);
}

}  // namespace

TEST(FluidSurfaceMesher, DeterministicByteIdentical) {
    const float spacing = 0.05f;
    const std::vector<Vec3> pool = SettledPool(8, 8, 6, spacing, {0.0f, 0.0f, 0.0f});
    const FluidSurfaceParams p = PoolParams(pool, spacing);

    const MeshGeometry a = MarchFluidSurface(pool, p);
    const MeshGeometry b = MarchFluidSurface(pool, p);

    PrintStats("determinism", Analyze(a));
    ASSERT_GT(a.VertexCount(), 0u);
    ASSERT_GT(a.TriangleCount(), 0u);

    ASSERT_EQ(a.positions.size(), b.positions.size());
    ASSERT_EQ(a.indices.size(), b.indices.size());
    ASSERT_EQ(a.normals.size(), b.normals.size());
    EXPECT_EQ(0, std::memcmp(a.positions.data(), b.positions.data(),
                             a.positions.size() * sizeof(float)));
    EXPECT_EQ(0, std::memcmp(a.indices.data(), b.indices.data(),
                             a.indices.size() * sizeof(uint32_t)));
    EXPECT_EQ(0, std::memcmp(a.normals.data(), b.normals.data(),
                             a.normals.size() * sizeof(float)));
    std::printf("    memcmp(positions/indices/normals) all == 0 (byte-identical)\n");
    EXPECT_TRUE(a.uvs.empty());
}

TEST(FluidSurfaceMesher, WatertightClosedBody) {
    const float spacing = 0.05f;
    const std::vector<Vec3> pool = SettledPool(8, 8, 6, spacing, {0.0f, 0.0f, 0.0f});
    const MeshGeometry m = MarchFluidSurface(pool, PoolParams(pool, spacing));
    const MeshStats s = Analyze(m);
    PrintStats("watertight", s);

    ASSERT_GT(s.tris, 0u);
    EXPECT_EQ(1, s.components) << "settled pool should be ONE connected body, not N blobs";
    EXPECT_EQ(0, s.nonmanifold_edges) << "every edge shared by <=2 triangles";
    EXPECT_EQ(0, s.boundary_edges) << "a closed mesh has no once-used (hole-rim) edges";
}

TEST(FluidSurfaceMesher, TracksParticleEnvelope) {
    const float spacing = 0.05f;
    const Vec3 corner{0.1f, -0.2f, 0.3f};
    const std::vector<Vec3> pool = SettledPool(8, 8, 6, spacing, corner);
    const FluidSurfaceParams p = PoolParams(pool, spacing);
    const MeshGeometry m = MarchFluidSurface(pool, p);
    const MeshStats s = Analyze(m);
    PrintStats("envelope", s);

    // Particle AABB.
    Vec3 plo = pool[0], phi = pool[0];
    for (const auto& q : pool) {
        plo.x = std::min(plo.x, q.x); plo.y = std::min(plo.y, q.y); plo.z = std::min(plo.z, q.z);
        phi.x = std::max(phi.x, q.x); phi.y = std::max(phi.y, q.y); phi.z = std::max(phi.z, q.z);
    }
    std::printf("    particle bbox=[%.4f,%.4f,%.4f]..[%.4f,%.4f,%.4f] (h=%.4f)\n",
                plo.x, plo.y, plo.z, phi.x, phi.y, phi.z, p.h);

    // The surface surrounds the particles (mesh lo below particle lo, mesh hi above)
    // and does not balloon further than ~h past them.
    const float tol = 1.5f * p.h;
    EXPECT_LE(s.lo.x, plo.x); EXPECT_LE(s.lo.y, plo.y); EXPECT_LE(s.lo.z, plo.z);
    EXPECT_GE(s.hi.x, phi.x); EXPECT_GE(s.hi.y, phi.y); EXPECT_GE(s.hi.z, phi.z);
    EXPECT_GE(s.lo.x, plo.x - tol); EXPECT_GE(s.lo.y, plo.y - tol); EXPECT_GE(s.lo.z, plo.z - tol);
    EXPECT_LE(s.hi.x, phi.x + tol); EXPECT_LE(s.hi.y, phi.y + tol); EXPECT_LE(s.hi.z, phi.z + tol);
}

TEST(FluidSurfaceMesher, NormalsFiniteAndUnit) {
    const float spacing = 0.05f;
    const std::vector<Vec3> pool = SettledPool(8, 8, 6, spacing, {0.0f, 0.0f, 0.0f});
    const MeshGeometry m = MarchFluidSurface(pool, PoolParams(pool, spacing));
    const MeshStats s = Analyze(m);
    PrintStats("normals", s);

    ASSERT_GT(s.verts, 0u);
    EXPECT_TRUE(s.all_finite_unit) << "every normal must be finite + unit length";
    EXPECT_NEAR(s.min_nlen, 1.0f, 1e-3f);
    EXPECT_NEAR(s.max_nlen, 1.0f, 1e-3f);

    // Outward winding: the area-weighted geometric normal of each triangle should
    // broadly agree with the averaged gradient (vertex) normal (positive dot).
    int agree = 0, total = 0;
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        const uint32_t i0 = m.indices[t], i1 = m.indices[t + 1], i2 = m.indices[t + 2];
        const Vec3 p0{m.positions[i0 * 3], m.positions[i0 * 3 + 1], m.positions[i0 * 3 + 2]};
        const Vec3 p1{m.positions[i1 * 3], m.positions[i1 * 3 + 1], m.positions[i1 * 3 + 2]};
        const Vec3 p2{m.positions[i2 * 3], m.positions[i2 * 3 + 1], m.positions[i2 * 3 + 2]};
        const Vec3 gn = (p1 - p0).Cross(p2 - p0);
        Vec3 vn{0, 0, 0};
        for (uint32_t v : {i0, i1, i2}) vn += Vec3{m.normals[v * 3], m.normals[v * 3 + 1], m.normals[v * 3 + 2]};
        if (gn.Dot(vn) > 0.0f) ++agree;
        ++total;
    }
    std::printf("    outward-winding agreement = %d / %d triangles\n", agree, total);
    EXPECT_GT(agree, total * 9 / 10) << "geometric winding should match gradient normals (outward)";
}

TEST(FluidSurfaceMesher, SphereOfParticlesIsRoughlySpherical) {
    const float spacing = 0.04f;
    const Vec3 center{0.5f, 0.5f, 0.5f};
    const float radius = 0.16f;
    const std::vector<Vec3> ball = ParticleBall(center, radius, spacing);
    FluidSurfaceParams p = PoolParams(ball, spacing);
    const MeshGeometry m = MarchFluidSurface(ball, p);
    const MeshStats s = Analyze(m);
    PrintStats("sphere", s);
    std::printf("    input particles=%zu radius=%.3f\n", ball.size(), radius);

    ASSERT_GT(s.tris, 0u);
    EXPECT_EQ(1, s.components);
    EXPECT_EQ(0, s.boundary_edges);

    // Vertex radii should cluster around a single shell radius (low rel. spread).
    double sum = 0.0, sum2 = 0.0;
    double rmin = 1e30, rmax = 0.0;
    for (uint32_t v = 0; v < s.verts; ++v) {
        const Vec3 d{m.positions[v * 3] - center.x, m.positions[v * 3 + 1] - center.y,
                     m.positions[v * 3 + 2] - center.z};
        const double r = std::sqrt(d.Dot(d));
        sum += r; sum2 += r * r; rmin = std::min(rmin, r); rmax = std::max(rmax, r);
    }
    const double mean = sum / s.verts;
    const double var = sum2 / s.verts - mean * mean;
    const double stdev = std::sqrt(var > 0.0 ? var : 0.0);
    std::printf("    vertex radius mean=%.4f stdev=%.4f (rel=%.3f) range=[%.4f,%.4f]\n",
                mean, stdev, stdev / mean, rmin, rmax);
    EXPECT_LT(stdev / mean, 0.20) << "a sphere of particles should mesh to a thin spherical shell";
}
