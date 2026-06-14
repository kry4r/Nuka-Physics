// ---------------------------------------------------------------------------
// v0.7 p13 GATE: the self-written CUDA SENSOR renderer (camera -> LBVH ->
// per-primitive closest-hit (triangle/sphere/sparse-SDF) -> Lambert+GGX shade +
// hard shadows -> full AOV framebuffer) must:
//   * match a CPU brute-force ray-vs-all-prims oracle that REUSES the same
//     __host__ __device__ intersection + shade functions (triangle hit + bary
//     EXACT, sphere depth + normal EXACT, SDF hit within a stated tolerance,
//     Lambert/GGX == analytic at chosen pixels, hard-shadow occlusion correct),
//   * be NaN-free for a surface-origin axis-parallel (shadow) ray with correct
//     occlusion (the live 0*inf slab guard),
//   * be two-run byte-exact across ALL AOV buffers (D1).
//
// Built with -ffp-contract=off to pair with the kernel's --fmad=false so the
// shared fp64-internal intersection arithmetic is host==device bit-for-bit.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "import/cooker/sparse_sdf_cooker.hpp"
#include "math/vec3.hpp"
#include "rt/camera.hpp"
#include "rt/framebuffer.hpp"
#include "phi/backend_cuda/rt/intersect_primitives.cuh" // host-callable shared intersections
#include "rt/material.hpp"
#include "phi/backend_cuda/rt/ray_box.cuh"              // host-callable: kNoPrim/RtMissDepth/RtClosestHitUpdate/RayBoxIntersect
#include "rt/scene_render.hpp"
#include "phi/backend_cuda/rt/shading.cuh"              // host-callable ShadeDirect
#include "runtime/sdf/sparse_sdf_query.cuh"
#include "tests/import/sdf_test_meshes.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace nuka;

namespace {

constexpr float kPi = 3.14159265358979323846f;

// ---- Host oracle: brute-force ray-vs-all-prims, sharing the device's exact
// __host__ __device__ intersection + closest-hit + shade. SDF arrays stay in
// host memory; we build a host SparseSdfDevice view per SDF prim. ------------
struct HostScene {
    const rt::Scene* scene;
    // Host SDF views (cell arrays in host memory + math::Vec3 gradients).
    std::vector<std::vector<math::Vec3>> sdf_grads;
    std::vector<runtime::sdf::SparseSdfDevice> sdf_hdrs;
};

HostScene MakeHostScene(const rt::Scene& scene) {
    HostScene hs;
    hs.scene = &scene;
    hs.sdf_grads.resize(scene.sdfs.size());
    hs.sdf_hdrs.resize(scene.sdfs.size());
    for (size_t i = 0; i < scene.sdfs.size(); ++i) {
        const auto& sd = scene.sdfs[i];
        hs.sdf_grads[i] = sd.gradients;
        runtime::sdf::SparseSdfDevice hdr = sd.header;
        hdr.cell_keys = sd.keys.data();
        hdr.cell_values = sd.values.data();
        hdr.cell_gradients = hs.sdf_grads[i].data();
        hdr.cell_count = static_cast<uint32_t>(sd.values.size());
        hs.sdf_hdrs[i] = hdr;
    }
    return hs;
}

// flat prim id -> (kind, sub, material) in declaration order: tri, sph, sdf.
struct OraclePrim { uint32_t kind; uint32_t sub; uint32_t material_id; };

std::vector<OraclePrim> FlatPrims(const rt::Scene& s) {
    std::vector<OraclePrim> p;
    for (const auto& t : s.triangles)
        p.push_back({0u, static_cast<uint32_t>(&t - s.triangles.data()), t.material_id});
    for (const auto& sp : s.spheres)
        p.push_back({1u, static_cast<uint32_t>(&sp - s.spheres.data()), sp.material_id});
    for (const auto& sd : s.sdfs)
        p.push_back({2u, static_cast<uint32_t>(&sd - s.sdfs.data()), sd.material_id});
    return p;
}

bool OracleIntersectT(const HostScene& hs, const OraclePrim& p,
                      const math::Vec3& o, const math::Vec3& d, float t_min,
                      float* out_t) {
    const auto& s = *hs.scene;
    if (p.kind == 0u) {
        float u, v; math::Vec3 n;
        return rt::RayTriangleIntersect(o, d, s.triangles[p.sub].v0,
                                        s.triangles[p.sub].v1, s.triangles[p.sub].v2,
                                        t_min, out_t, &u, &v, &n);
    } else if (p.kind == 1u) {
        math::Vec3 n; float u, v;
        return rt::RaySphereIntersect(o, d, s.spheres[p.sub].center,
                                      s.spheres[p.sub].radius, t_min, out_t, &n, &u, &v);
    } else {
        math::Vec3 n;
        return rt::RaySdfSphereTrace(o, d, hs.sdf_hdrs[p.sub], s.sdfs[p.sub].aabb,
                                     t_min, s.sdfs[p.sub].surface_eps,
                                     s.sdfs[p.sub].max_iters, out_t, &n);
    }
}

// Closest-hit over all prims (lowest-index tie-break == RtClosestHitUpdate).
bool OracleClosestHit(const HostScene& hs, const std::vector<OraclePrim>& prims,
                      const math::Vec3& o, const math::Vec3& d, float t_min,
                      float* best_t, uint32_t* best_prim) {
    *best_t = rt::RtMissDepth();
    *best_prim = rt::kNoPrim;
    for (uint32_t i = 0; i < prims.size(); ++i) {
        float t;
        if (OracleIntersectT(hs, prims[i], o, d, t_min, &t)) {
            if (t >= t_min) {
                rt::RtClosestHitUpdate(t, i, best_t, best_prim);
            }
        }
    }
    return *best_prim != rt::kNoPrim;
}

void OracleReconstruct(const HostScene& hs, const std::vector<OraclePrim>& prims,
                       uint32_t prim, const math::Vec3& o, const math::Vec3& d,
                       math::Vec3* n, float* u, float* v, uint32_t* mat) {
    const auto& s = *hs.scene;
    const OraclePrim p = prims[prim];
    *mat = p.material_id;
    if (p.kind == 0u) {
        float t;
        rt::RayTriangleIntersect(o, d, s.triangles[p.sub].v0, s.triangles[p.sub].v1,
                                 s.triangles[p.sub].v2, 0.0f, &t, u, v, n);
    } else if (p.kind == 1u) {
        float t;
        rt::RaySphereIntersect(o, d, s.spheres[p.sub].center, s.spheres[p.sub].radius,
                               0.0f, &t, n, u, v);
    } else {
        float t;
        rt::RaySdfSphereTrace(o, d, hs.sdf_hdrs[p.sub], s.sdfs[p.sub].aabb, 0.0f,
                              s.sdfs[p.sub].surface_eps, s.sdfs[p.sub].max_iters, &t, n);
        *u = 0.0f; *v = 0.0f;
    }
}

// Full host oracle render -> a Framebuffer mirroring RenderScene's AOVs.
rt::Framebuffer OracleRender(const rt::Scene& scene, const rt::PinholeCamera& cam) {
    HostScene hs = MakeHostScene(scene);
    const auto prims = FlatPrims(scene);
    rt::Framebuffer fb;
    fb.width = cam.width; fb.height = cam.height;
    const size_t px = fb.Pixels();
    fb.color.assign(px * 3, 0.0f);
    fb.depth.assign(px, rt::RtMissDepth());
    fb.normal.assign(px * 3, 0.0f);
    fb.albedo.assign(px * 3, 0.0f);
    fb.uv.assign(px * 2, 0.0f);
    fb.prim.assign(px, rt::kNoPrim);

    for (uint32_t y = 0; y < cam.height; ++y) {
        for (uint32_t x = 0; x < cam.width; ++x) {
            const rt::Ray ray = cam.GenerateRay(x, y);
            float bt; uint32_t bp;
            if (!OracleClosestHit(hs, prims, ray.origin, ray.dir, 0.0f, &bt, &bp)) {
                continue;
            }
            const size_t p = static_cast<size_t>(y) * cam.width + x;
            fb.depth[p] = bt;
            fb.prim[p] = bp;
            math::Vec3 n; float u, v; uint32_t mat;
            OracleReconstruct(hs, prims, bp, ray.origin, ray.dir, &n, &u, &v, &mat);
            fb.normal[p * 3 + 0] = n.x; fb.normal[p * 3 + 1] = n.y; fb.normal[p * 3 + 2] = n.z;
            fb.uv[p * 2 + 0] = u; fb.uv[p * 2 + 1] = v;
            const rt::Material m = scene.materials[mat];
            fb.albedo[p * 3 + 0] = m.albedo.x; fb.albedo[p * 3 + 1] = m.albedo.y; fb.albedo[p * 3 + 2] = m.albedo.z;

            const math::Vec3 hit{
                static_cast<float>(static_cast<double>(ray.origin.x) + static_cast<double>(bt) * ray.dir.x),
                static_cast<float>(static_cast<double>(ray.origin.y) + static_cast<double>(bt) * ray.dir.y),
                static_cast<float>(static_cast<double>(ray.origin.z) + static_cast<double>(bt) * ray.dir.z)};
            const math::Vec3 V = rt::RtNormalize(math::Vec3{-ray.dir.x, -ray.dir.y, -ray.dir.z});
            math::Vec3 L; float light_dist;
            if (scene.light.directional) {
                L = rt::RtNormalize(math::Vec3{-scene.light.direction.x, -scene.light.direction.y, -scene.light.direction.z});
                light_dist = rt::RtMissDepth();
            } else {
                const math::Vec3 to{scene.light.position.x - hit.x, scene.light.position.y - hit.y, scene.light.position.z - hit.z};
                L = rt::RtNormalize(to);
                light_dist = static_cast<float>(std::sqrt(double(to.x)*to.x + double(to.y)*to.y + double(to.z)*to.z));
            }
            math::Vec3 ns = n;
            const double nv = double(n.x)*V.x + double(n.y)*V.y + double(n.z)*V.z;
            if (nv < 0.0) ns = math::Vec3{-n.x, -n.y, -n.z};
            const float seps = 1.0e-3f;
            const math::Vec3 so{hit.x + ns.x*seps, hit.y + ns.y*seps, hit.z + ns.z*seps};
            int lit = 1; float st; uint32_t sp;
            if (OracleClosestHit(hs, prims, so, L, seps, &st, &sp)) {
                if (sp != rt::kNoPrim && st < light_dist - seps) lit = 0;
            }
            const math::Vec3 lc{scene.light.color.x*scene.light.intensity,
                                scene.light.color.y*scene.light.intensity,
                                scene.light.color.z*scene.light.intensity};
            const math::Vec3 c = rt::ShadeDirect(n, V, L, lc, m, lit, scene.ambient.color);
            fb.color[p*3+0]=c.x; fb.color[p*3+1]=c.y; fb.color[p*3+2]=c.z;
        }
    }
    return fb;
}

// A quad (2 triangles) facing -Z at z=z0, spanning [-h,h]^2 in x/y, with a
// material id. Wound so the geometric normal is +Z (toward a camera at origin
// looking +Z). v0,v1,v2 order chosen so cross(e1,e2) = +Z.
void AddQuad(rt::Scene& s, float z0, float h, uint32_t mat) {
    const math::Vec3 a{-h, -h, z0}, b{h, -h, z0}, c{h, h, z0}, d{-h, h, z0};
    s.triangles.push_back({a, b, c, mat});
    s.triangles.push_back({a, c, d, mat});
}

} // namespace

// =====================================================================
// 0. GROUND-TRUTH ANCHOR: the oracle SHARES the kernel's Moller-Trumbore, so the
//    "exact vs oracle" gates prove host==device, not host==truth. Anchor the
//    intersection fns directly against HAND-COMPUTED values so a symmetric bug
//    (u/v swap, bary scale, t off by 1/a) cannot pass silently.
// =====================================================================
TEST(RtSceneRender, IntersectFunctionsMatchHandComputedTruth) {
    // Triangle: v0=(0,0,5), v1=(2,0,5), v2=(0,2,5); ray o=(0,0,0) toward
    // (0.5,0.5,5). The plane is z=5, hit point P=(0.5,0.5,5):
    //   P = v0 + u*(v1-v0) + v*(v2-v0) = (2u, 2v, 5) -> u=0.25, v=0.25.
    //   t (along the UNIT dir) = 5/dir.z = 5.0497524691810...
    //   geometric normal = normalize(cross((2,0,0),(0,2,0))) = (0,0,1).
    {
        const math::Vec3 v0{0,0,5}, v1{2,0,5}, v2{0,2,5};
        const math::Vec3 o{0,0,0};
        const math::Vec3 dir = rt::RtNormalize(math::Vec3{0.5f, 0.5f, 5.0f});
        float t, u, v; math::Vec3 n;
        const bool hit = rt::RayTriangleIntersect(o, dir, v0, v1, v2, 0.0f, &t, &u, &v, &n);
        ASSERT_TRUE(hit);
        EXPECT_NEAR(u, 0.25f, 1e-5f);
        EXPECT_NEAR(v, 0.25f, 1e-5f);
        EXPECT_NEAR(t, 5.0497524691810f, 1e-4f);
        EXPECT_NEAR(n.x, 0.0f, 1e-6f);
        EXPECT_NEAR(n.y, 0.0f, 1e-6f);
        EXPECT_NEAR(std::fabs(n.z), 1.0f, 1e-6f);
    }
    // Sphere: center (0,0,5), r=1.5; ray o=(0,0,0) toward +Z hits the front face
    // at z=3.5 -> t=3.5, outward normal at the hit = (0,0,-1).
    {
        const math::Vec3 o{0,0,0}, d{0,0,1}, c{0,0,5};
        float t, u, v; math::Vec3 n;
        const bool hit = rt::RaySphereIntersect(o, d, c, 1.5f, 0.0f, &t, &n, &u, &v);
        ASSERT_TRUE(hit);
        EXPECT_NEAR(t, 3.5f, 1e-5f);
        EXPECT_NEAR(n.x, 0.0f, 1e-6f);
        EXPECT_NEAR(n.y, 0.0f, 1e-6f);
        EXPECT_NEAR(n.z, -1.0f, 1e-6f);
    }
}

// =====================================================================
// 1. TRIANGLE: hit t + barycentric EXACT vs oracle (single triangle, center).
// =====================================================================
TEST(RtSceneRender, TriangleHitAndBarycentricExact) {
    rt::Scene scene;
    // A single large triangle facing the camera at z=3, filling much of the frame.
    scene.triangles.push_back({{-4.0f, -3.0f, 3.0f}, {4.0f, -3.0f, 3.0f}, {0.0f, 5.0f, 3.0f}, 0u});
    scene.materials.push_back(rt::Material{});
    scene.light.directional = true;
    scene.light.direction = {0.0f, 0.0f, 1.0f};

    auto cam = rt::BuildPinhole({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.6f * kPi, 64u, 48u);
    auto got = rt::RenderScene(scene, cam);
    auto ora = OracleRender(scene, cam);

    float max_depth_err = 0.0f, max_uv_err = 0.0f;
    size_t hits = 0, prim_mismatch = 0;
    for (size_t i = 0; i < got.depth.size(); ++i) {
        if (got.prim[i] != ora.prim[i]) ++prim_mismatch;
        if (got.prim[i] != rt::kNoPrim && ora.prim[i] != rt::kNoPrim) {
            ++hits;
            max_depth_err = std::max(max_depth_err, std::fabs(got.depth[i] - ora.depth[i]));
            max_uv_err = std::max(max_uv_err, std::fabs(got.uv[2*i] - ora.uv[2*i]));
            max_uv_err = std::max(max_uv_err, std::fabs(got.uv[2*i+1] - ora.uv[2*i+1]));
        }
    }
    std::printf("[triangle] hits=%zu prim_mismatch=%zu depth_err=%g uv_err=%g\n",
                hits, prim_mismatch, max_depth_err, max_uv_err);
    EXPECT_GT(hits, 100u);
    EXPECT_EQ(prim_mismatch, 0u);
    EXPECT_EQ(max_depth_err, 0.0f);   // EXACT (shared fp64 intersection)
    EXPECT_EQ(max_uv_err, 0.0f);      // barycentric EXACT
}

// =====================================================================
// 2. SPHERE: depth + normal EXACT vs oracle.
// =====================================================================
TEST(RtSceneRender, SphereDepthAndNormalExact) {
    rt::Scene scene;
    scene.spheres.push_back({{0.0f, 0.0f, 5.0f}, 1.5f, 0u});
    scene.materials.push_back(rt::Material{});
    scene.light.directional = true;
    scene.light.direction = {0.0f, 0.0f, 1.0f};

    auto cam = rt::BuildPinhole({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.6f * kPi, 80u, 80u);
    auto got = rt::RenderScene(scene, cam);
    auto ora = OracleRender(scene, cam);

    float max_depth_err = 0.0f, max_normal_err = 0.0f;
    size_t hits = 0, prim_mismatch = 0;
    for (size_t i = 0; i < got.depth.size(); ++i) {
        if (got.prim[i] != ora.prim[i]) ++prim_mismatch;
        if (got.prim[i] != rt::kNoPrim && ora.prim[i] != rt::kNoPrim) {
            ++hits;
            max_depth_err = std::max(max_depth_err, std::fabs(got.depth[i] - ora.depth[i]));
            for (int k = 0; k < 3; ++k)
                max_normal_err = std::max(max_normal_err, std::fabs(got.normal[3*i+k] - ora.normal[3*i+k]));
        }
    }
    std::printf("[sphere] hits=%zu prim_mismatch=%zu depth_err=%g normal_err=%g\n",
                hits, prim_mismatch, max_depth_err, max_normal_err);
    EXPECT_GT(hits, 100u);
    EXPECT_EQ(prim_mismatch, 0u);
    EXPECT_EQ(max_depth_err, 0.0f);
    EXPECT_EQ(max_normal_err, 0.0f);

    // Analytic depth at center pixel: ray +z hits sphere front at z=center.z-r=3.5.
    const uint32_t cx = cam.width/2, cy = cam.height/2;
    // (even res -> use oracle agreement already proven; spot-check a near-center hit)
    const size_t cp = static_cast<size_t>(cy)*cam.width + cx;
    EXPECT_NE(got.prim[cp], rt::kNoPrim);
    EXPECT_NEAR(got.depth[cp], 3.5f, 0.05f);
}

// =====================================================================
// 3. SPARSE-SDF: sphere-trace hit within tolerance vs the analytic sphere.
// =====================================================================
TEST(RtSceneRender, SdfSphereTraceHitWithinTolerance) {
    using nuka::import::cooker::CookSparseSdf;
    using nuka::import::cooker::SparseSdfParams;
    // Cook a unit sphere SDF at the origin's local frame; place it at identity.
    const auto mesh = nuka::test::SphereMesh(0.0f, 0.0f, 0.0f, 1.0f, 48, 96);
    SparseSdfParams params; params.voxel_size = 0.05f; params.band_voxels = 6u;
    const auto cooked = CookSparseSdf(mesh.vertices.data(),
                                      static_cast<uint32_t>(mesh.vertices.size()/3),
                                      mesh.indices.data(),
                                      static_cast<uint32_t>(mesh.indices.size()/3),
                                      params);
    ASSERT_GT(cooked.CellCount(), 0u);

    rt::Scene scene;
    rt::SdfPrim sp;
    sp.header.origin = {cooked.origin[0], cooked.origin[1], cooked.origin[2]};
    sp.header.voxel_size = cooked.voxel_size;
    sp.header.dims[0] = cooked.dims[0]; sp.header.dims[1] = cooked.dims[1]; sp.header.dims[2] = cooked.dims[2];
    sp.keys = cooked.cell_keys;
    sp.values = cooked.cell_values;
    sp.gradients.resize(cooked.CellCount());
    for (uint32_t c = 0; c < cooked.CellCount(); ++c)
        sp.gradients[c] = {cooked.cell_gradients[3*c], cooked.cell_gradients[3*c+1], cooked.cell_gradients[3*c+2]};
    // World AABB: the sphere is within [-1.2,1.2]^3; clip the march here.
    sp.aabb.min = {-1.3f, -1.3f, -1.3f};
    sp.aabb.max = {1.3f, 1.3f, 1.3f};
    sp.surface_eps = 0.01f;
    sp.max_iters = 512;
    sp.material_id = 0u;
    scene.sdfs.push_back(sp);
    scene.materials.push_back(rt::Material{});
    scene.light.directional = true;
    scene.light.direction = {0.0f, 0.0f, 1.0f};

    // Camera looking down +z toward the sphere at origin from z=-4.
    auto cam = rt::BuildPinhole({0.0f, 0.0f, -4.0f}, {0.0f, 0.0f, 0.0f},
                                {0.0f, 1.0f, 0.0f}, 0.5f * kPi, 64u, 64u);
    auto got = rt::RenderScene(scene, cam);

    // Oracle agreement (device vs host sphere-trace) -- both iterative, should be
    // bit-identical (same shared fn, --fmad/-ffp-contract paired).
    auto ora = OracleRender(scene, cam);
    float max_depth_err = 0.0f; size_t hits = 0, prim_mismatch = 0;
    for (size_t i = 0; i < got.depth.size(); ++i) {
        if (got.prim[i] != ora.prim[i]) ++prim_mismatch;
        if (got.prim[i] != rt::kNoPrim && ora.prim[i] != rt::kNoPrim) {
            ++hits;
            max_depth_err = std::max(max_depth_err, std::fabs(got.depth[i] - ora.depth[i]));
        }
    }
    std::printf("[sdf] hits=%zu prim_mismatch=%zu device_vs_oracle_depth_err=%g\n",
                hits, prim_mismatch, max_depth_err);
    EXPECT_GT(hits, 100u);
    EXPECT_EQ(prim_mismatch, 0u);
    EXPECT_EQ(max_depth_err, 0.0f); // device == host sphere-trace (shared fn)

    // Analytic tolerance: the center ray (from z=-4 toward +z) hits the sphere
    // front face at z=-1, i.e. depth = 3.0. The sphere-trace terminates within
    // surface_eps + voxel of the true surface.
    const uint32_t cx = cam.width/2, cy = cam.height/2;
    const size_t cp = static_cast<size_t>(cy)*cam.width + cx;
    ASSERT_NE(got.prim[cp], rt::kNoPrim);
    const float analytic = 3.0f; // |(-4) - (-1)|
    const float err = std::fabs(got.depth[cp] - analytic);
    const float tol = sp.surface_eps + cooked.voxel_size; // stated tolerance
    std::printf("[sdf] center depth=%g analytic=%g err=%g tol=%g\n",
                got.depth[cp], analytic, err, tol);
    EXPECT_LT(err, tol);
}

// =====================================================================
// 4. SHADING: Lambert (+GGX) == analytic at a chosen lit pixel. A single quad
//    facing the camera, directional light along -Z (straight back at the quad),
//    so N.L == 1 at the front face -> color == albedo*(1-metallic)/pi + ambient*albedo.
// =====================================================================
TEST(RtSceneRender, LambertShadingMatchesAnalytic) {
    rt::Scene scene;
    rt::Material m; m.albedo = {0.6f, 0.4f, 0.2f}; m.metallic = 0.0f; m.roughness = 0.5f;
    scene.materials.push_back(m);
    AddQuad(scene, 5.0f, 3.0f, 0u);
    // Directional light travels +Z (away from camera, into the quad's front),
    // so L (toward light) = -Z, and the front-facing (viewer-faced) normal is -Z
    // after flip => N.L = 1.
    scene.light.directional = true;
    scene.light.direction = {0.0f, 0.0f, 1.0f};
    scene.light.color = {1.0f, 1.0f, 1.0f};
    scene.light.intensity = 1.0f;
    scene.ambient.color = {0.03f, 0.03f, 0.03f};

    auto cam = rt::BuildPinhole({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.3f * kPi, 65u, 65u);
    auto got = rt::RenderScene(scene, cam);
    const uint32_t cx = 32, cy = 32;            // dead-center pixel (odd res)
    const size_t cp = static_cast<size_t>(cy)*cam.width + cx;
    ASSERT_NE(got.prim[cp], rt::kNoPrim);

    // Analytic: center ray goes +Z, hits the quad front. V = -ray.dir = -Z.
    // N (geometric, +Z) faced toward V (-Z) -> Nf = -Z. L (toward light) = -Z.
    // NoL = 1. metallic 0 -> spec 0. color = light*albedo*(1/pi)*1 + ambient*albedo.
    const float inv_pi = 1.0f / kPi;
    const float exp_r = 1.0f * 0.6f * inv_pi * 1.0f + 0.03f * 0.6f;
    const float exp_g = 1.0f * 0.4f * inv_pi * 1.0f + 0.03f * 0.4f;
    const float exp_b = 1.0f * 0.2f * inv_pi * 1.0f + 0.03f * 0.2f;
    std::printf("[lambert] got=(%g,%g,%g) exp=(%g,%g,%g)\n",
                got.color[cp*3], got.color[cp*3+1], got.color[cp*3+2], exp_r, exp_g, exp_b);
    EXPECT_NEAR(got.color[cp*3+0], exp_r, 1e-5f);
    EXPECT_NEAR(got.color[cp*3+1], exp_g, 1e-5f);
    EXPECT_NEAR(got.color[cp*3+2], exp_b, 1e-5f);

    // And the full color buffer matches the oracle bit-for-bit.
    auto ora = OracleRender(scene, cam);
    EXPECT_EQ(0, std::memcmp(got.color.data(), ora.color.data(),
                             got.color.size()*sizeof(float)))
        << "color buffer diverged from shading oracle";
}

// =====================================================================
// 4b. GGX SPECULAR == known BRDF value. Same head-on quad, but metallic=1 so the
//     diffuse term vanishes and the color is pure GGX D. At the center pixel
//     V=L=-Z and the viewer-faced normal is -Z, so the half vector H=-Z and
//     NoH=1. With alpha = roughness^2: D(NoH=1) = a^2/(pi*(a^2)^2) = 1/(pi*a^2)
//     = 1/(pi*roughness^4). color = light * D * NoL(=1) + ambient*albedo.
// =====================================================================
TEST(RtSceneRender, GgxSpecularMatchesKnownBrdf) {
    rt::Scene scene;
    rt::Material m; m.albedo = {0.5f, 0.5f, 0.5f}; m.metallic = 1.0f; m.roughness = 0.4f;
    scene.materials.push_back(m);
    AddQuad(scene, 5.0f, 3.0f, 0u);
    scene.light.directional = true;
    scene.light.direction = {0.0f, 0.0f, 1.0f};
    scene.light.color = {1.0f, 1.0f, 1.0f};
    scene.light.intensity = 1.0f;
    scene.ambient.color = {0.03f, 0.03f, 0.03f};

    auto cam = rt::BuildPinhole({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.3f * kPi, 65u, 65u);
    auto got = rt::RenderScene(scene, cam);
    const size_t cp = static_cast<size_t>(32)*cam.width + 32;
    ASSERT_NE(got.prim[cp], rt::kNoPrim);

    const double a = 0.4 * 0.4;                 // alpha = roughness^2
    const double D = a * a / (kPi * a * a * a * a); // = 1/(pi*a^2)
    const float exp_r = static_cast<float>(1.0 * D * 1.0 + 0.03 * 0.5);
    std::printf("[ggx] got_r=%g exp_r=%g (D=%g)\n", got.color[cp*3], exp_r, D);
    EXPECT_NEAR(got.color[cp*3+0], exp_r, 1e-4f);

    auto ora = OracleRender(scene, cam);
    EXPECT_EQ(0, std::memcmp(got.color.data(), ora.color.data(),
                             got.color.size()*sizeof(float)));
}

// =====================================================================
// 5. HARD SHADOWS: a small occluder between a back wall and the light casts a
//    shadow; shadowed pixels are darker (ambient-only) than lit pixels.
// =====================================================================
TEST(RtSceneRender, HardShadowOcclusion) {
    rt::Scene scene;
    rt::Material wall; wall.albedo = {0.8f, 0.8f, 0.8f};
    scene.materials.push_back(wall);            // mat 0: wall
    rt::Material occ; occ.albedo = {0.2f, 0.2f, 0.2f};
    scene.materials.push_back(occ);             // mat 1: occluder

    // Back wall at z=8, large. Occluder quad at z=5, small, between wall + light.
    AddQuad(scene, 8.0f, 4.0f, 0u);
    AddQuad(scene, 5.0f, 0.6f, 1u);

    // Point light in front-left so the occluder shadows part of the wall.
    scene.light.directional = false;
    scene.light.position = {-3.0f, 0.0f, 2.0f};
    scene.light.color = {1.0f, 1.0f, 1.0f};
    scene.light.intensity = 5.0f;
    scene.ambient.color = {0.02f, 0.02f, 0.02f};

    auto cam = rt::BuildPinhole({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.6f * kPi, 96u, 96u);
    auto got = rt::RenderScene(scene, cam);
    auto ora = OracleRender(scene, cam);

    // Oracle agreement on the whole image (color memcmp -- shadow logic shared).
    EXPECT_EQ(0, std::memcmp(got.color.data(), ora.color.data(),
                             got.color.size()*sizeof(float)))
        << "shadowed render diverged from oracle";
    EXPECT_EQ(0, std::memcmp(got.prim.data(), ora.prim.data(),
                             got.prim.size()*sizeof(uint32_t)));

    // Count wall pixels that are lit (brighter) vs shadowed (ambient-only).
    size_t wall_lit = 0, wall_shadow = 0;
    for (size_t i = 0; i < got.prim.size(); ++i) {
        if (got.prim[i] == 0u) { // a wall pixel (mat 0 -> tri prim 0 or 1)
            const float lum = got.color[3*i] + got.color[3*i+1] + got.color[3*i+2];
            // Ambient-only luminance for the wall = 3 * 0.02 * 0.8 = 0.048.
            if (lum > 0.1f) ++wall_lit; else ++wall_shadow;
        }
    }
    std::printf("[shadow] wall_lit=%zu wall_shadow=%zu\n", wall_lit, wall_shadow);
    EXPECT_GT(wall_lit, 0u) << "no lit wall pixels -- light mis-aimed";
    EXPECT_GT(wall_shadow, 0u) << "no shadowed wall pixels -- occluder casts no shadow";
}

// =====================================================================
// 6. NaN GUARD: a SURFACE-ORIGIN AXIS-PARALLEL ray (the live 0*inf slab bug).
//    Fire a shadow-like ray that originates ON a box-aligned surface and travels
//    exactly axis-parallel, grazing another primitive's plane. Assert no NaN in
//    ANY AOV + correct occlusion. We construct a scene where the analytic shadow
//    answer is known and the shadow ray is axis-parallel.
// =====================================================================
TEST(RtSceneRender, NanGuardSurfaceOriginAxisParallelRay) {
    // Direct unit test of the shared slab math: a ray ON a plane, axis-parallel.
    {
        collision::AABB box; box.min = {0.0f, 0.0f, 0.0f}; box.max = {1.0f, 1.0f, 1.0f};
        // Origin exactly on the y=0 plane of the box; direction +X (axis-parallel
        // to the y/z slabs while sitting on the y-min plane => 0*inf risk).
        const math::Vec3 o{-1.0f, 0.0f, 0.5f};
        const math::Vec3 d{1.0f, 0.0f, 0.0f};
        float t;
        const bool hit = rt::RayBoxIntersect(o, d, box, &t);
        EXPECT_TRUE(hit);
        EXPECT_FALSE(std::isnan(t)) << "0*inf slab NaN not guarded";
        EXPECT_NEAR(t, 1.0f, 1e-5f); // enters the box at x=0 -> distance 1
    }

    // Scene-level: a wall + a thin occluder; a directional light exactly +X so
    // EVERY shadow ray is axis-parallel and the offset shadow origins land on
    // grazing planes. The render must be NaN-free and produce a shadow.
    rt::Scene scene;
    rt::Material wall; wall.albedo = {0.7f, 0.7f, 0.7f};
    scene.materials.push_back(wall);
    rt::Material occ; occ.albedo = {0.3f, 0.3f, 0.3f};
    scene.materials.push_back(occ);

    // Wall facing camera at z=6; an axis-aligned occluder quad in the XY plane
    // offset in +X so a +X shadow ray from the wall hits it.
    AddQuad(scene, 6.0f, 3.0f, 0u);
    // Occluder: a quad in the YZ plane at x=2 (normal +X), spanning z/y so a +X
    // shadow ray from wall points (x<2) toward the light is blocked.
    {
        const float x0 = 2.0f, h = 2.0f;
        const math::Vec3 a{x0, -h, 6.0f - h}, b{x0, -h, 6.0f + h},
                         c{x0, h, 6.0f + h}, dd{x0, h, 6.0f - h};
        scene.triangles.push_back({a, b, c, 1u});
        scene.triangles.push_back({a, c, dd, 1u});
    }
    scene.light.directional = true;
    scene.light.direction = {-1.0f, 0.0f, 0.0f}; // travels -X; L (to light) = +X
    scene.light.color = {1.0f, 1.0f, 1.0f};
    scene.light.intensity = 1.0f;
    scene.ambient.color = {0.05f, 0.05f, 0.05f};

    auto cam = rt::BuildPinhole({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.6f * kPi, 80u, 80u);
    auto got = rt::RenderScene(scene, cam);

    size_t nan_count = 0;
    for (size_t i = 0; i < got.color.size(); ++i) if (std::isnan(got.color[i])) ++nan_count;
    for (size_t i = 0; i < got.normal.size(); ++i) if (std::isnan(got.normal[i])) ++nan_count;
    for (size_t i = 0; i < got.depth.size(); ++i)
        if (std::isnan(got.depth[i])) ++nan_count; // +inf miss is fine, NaN is not
    EXPECT_EQ(nan_count, 0u) << "NaN leaked into an AOV (0*inf slab guard failed)";

    // Correctness: matches the oracle (which uses the SAME guarded math).
    auto ora = OracleRender(scene, cam);
    EXPECT_EQ(0, std::memcmp(got.color.data(), ora.color.data(),
                             got.color.size()*sizeof(float)));
    EXPECT_EQ(0, std::memcmp(got.prim.data(), ora.prim.data(),
                             got.prim.size()*sizeof(uint32_t)));

    // The +X-facing occluder must shadow some wall pixels at x<2.
    size_t shadowed = 0;
    for (uint32_t y = 0; y < cam.height; ++y)
        for (uint32_t x = 0; x < cam.width; ++x) {
            const size_t p = static_cast<size_t>(y)*cam.width + x;
            if (got.prim[p] == 0u || got.prim[p] == 1u) { // wall triangle
                const float lum = got.color[3*p]+got.color[3*p+1]+got.color[3*p+2];
                if (lum < 0.12f) ++shadowed; // ambient-only ~ 3*0.05*0.7=0.105
            }
        }
    std::printf("[nanguard] nan_count=%zu shadowed_wall=%zu\n", nan_count, shadowed);
    EXPECT_GT(shadowed, 0u) << "axis-parallel shadow produced no occlusion";
}

// =====================================================================
// 7. D1: two-run byte-exact across ALL AOV buffers (memcmp).
// =====================================================================
TEST(RtSceneRender, D1TwoRunByteExactAllAovs) {
    rt::Scene scene;
    rt::Material m0; m0.albedo = {0.7f, 0.5f, 0.3f}; m0.metallic = 0.3f; m0.roughness = 0.4f;
    rt::Material m1; m1.albedo = {0.2f, 0.6f, 0.8f}; m1.metallic = 0.0f; m1.roughness = 0.7f;
    scene.materials.push_back(m0);
    scene.materials.push_back(m1);
    // Mixed scene: triangles (a quad) + spheres.
    AddQuad(scene, 9.0f, 5.0f, 0u);
    for (int gx = -2; gx <= 2; ++gx)
        for (int gy = -2; gy <= 2; ++gy)
            scene.spheres.push_back({{gx*1.4f, gy*1.4f, 6.0f}, 0.5f, 1u});
    scene.light.directional = false;
    scene.light.position = {2.0f, 3.0f, 1.0f};
    scene.light.intensity = 8.0f;

    auto cam = rt::BuildPinhole({0.1f, -0.2f, 0.0f}, {0.0f, 0.0f, 6.0f},
                                {0.0f, 1.0f, 0.0f}, 0.8f * kPi, 128u, 96u);
    auto r1 = rt::RenderScene(scene, cam);
    auto r2 = rt::RenderScene(scene, cam);
    ASSERT_EQ(r1.color.size(), r2.color.size());
    EXPECT_EQ(0, std::memcmp(r1.color.data(), r2.color.data(), r1.color.size()*sizeof(float))) << "color not byte-exact";
    EXPECT_EQ(0, std::memcmp(r1.depth.data(), r2.depth.data(), r1.depth.size()*sizeof(float))) << "depth not byte-exact";
    EXPECT_EQ(0, std::memcmp(r1.normal.data(), r2.normal.data(), r1.normal.size()*sizeof(float))) << "normal not byte-exact";
    EXPECT_EQ(0, std::memcmp(r1.albedo.data(), r2.albedo.data(), r1.albedo.size()*sizeof(float))) << "albedo not byte-exact";
    EXPECT_EQ(0, std::memcmp(r1.uv.data(), r2.uv.data(), r1.uv.size()*sizeof(float))) << "uv not byte-exact";
    EXPECT_EQ(0, std::memcmp(r1.prim.data(), r2.prim.data(), r1.prim.size()*sizeof(uint32_t))) << "prim not byte-exact";
    std::printf("[d1] all 6 AOVs two-run byte-exact (%ux%u)\n", cam.width, cam.height);
}
