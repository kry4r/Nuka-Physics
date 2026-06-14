// ---------------------------------------------------------------------------
// v0.7 G1-A GATE: the self-written two-level (TLAS/BLAS) CUDA ray tracer
// (rt::BuildTwoLevelScene + rt::RenderFrame). The two-level tracer nests the p12
// stackless rt::TraverseRay two deep: a per-frame TLAS over instance world-AABBs
// whose leaf transforms the ray into each instance's local frame and runs an
// INNER traversal over that instance's once-built BLAS. It MUST:
//
//   Oracle 1 (D1 + host/device-parity + traversal-independence):
//     a NEW host two-level brute-force that mirrors the device TLAS-leaf math
//     EXACTLY (same hoisted QuatInvRotate, same shared intersect fns, same
//     PackPrimId) -> byte-exact host==device for ANY transform incl. rotation.
//     This shares the math, so it is NOT the primary correctness gate.
//
//   Oracle 2 (ALGORITHM-correctness, the trusted INDEPENDENT path):
//     bake each instance's transform into world-space vertices -> a flat p13
//     rt::Scene -> rt::RenderScene. Byte-exact (memcmp==0) color/depth/normal at
//     IDENTITY instances; near-exact (stated tolerance) for rotated/translated
//     instances (fp32 QuatInvRotate rounds the local ray differently than pre-
//     rounded flat world vertices). prim_id is NOT compared to Oracle 2 (flat ties
//     by global decl index, two-level by packed (inst,local)).
//
//   Analytic anchors (so the oracles are not circular):
//     p13's hand-computed single-triangle t, PLUS a NEW rotated+translated
//     instance whose world hit + t are hand-computed AND whose prim_id ==
//     PackPrimId(known_instance, known_local) is asserted exactly (the ONLY check
//     that exercises the bit-split + instance-high ordering).
//
//   Instance-transform tracking: move one instance -> an anchored background
//   pixel becomes that instance's color.
//
//   D1: render twice -> memcmp all 6 AOVs == 0.
//
// Built -ffp-contract=off to pair with the kernel's --fmad=false so Oracle 1's
// shared fp64-internal math (+ fp32 QuatInvRotate) is host==device bit-for-bit.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "import/cooker/sparse_sdf_cooker.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "rt/camera.hpp"
#include "rt/framebuffer.hpp"
#include "phi/backend_cuda/rt/instance_transform.cuh"   // host-callable QuatRotate/QuatInvRotate/TransformRayToLocal
#include "phi/backend_cuda/rt/intersect_primitives.cuh" // host-callable shared intersections
#include "rt/material.hpp"
#include "phi/backend_cuda/rt/prim_id.cuh"              // host-callable PackPrimId/UnpackPrimId
#include "phi/backend_cuda/rt/ray_box.cuh"             // host-callable kNoPrim/RtMissDepth/RtClosestHitUpdate
#include "rt/scene_render.hpp"         // flat p13 Scene + RenderScene (Oracle 2)
#include "phi/backend_cuda/rt/shading.cuh"             // host-callable ShadeDirect
#include "rt/two_level_render.hpp"
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

// ---- Oracle 1: host two-level brute-force. Mirrors the device TLAS-leaf math
// EXACTLY: for each instance, transform the ray to local with the SAME hoisted
// TransformRayToLocal, intersect the SAME shared fns, pack with the SAME
// PackPrimId. Triangle/sphere only (the anchors + parity scenes use those). ----
struct OracleHit {
    float best_t = rt::RtMissDepth();
    uint32_t best_prim = rt::kNoPrim;
};

// Build a HOST SparseSdfDevice view (cell arrays in host memory) from a SdfPrim,
// so the oracle can sphere-trace it with the SAME shared RaySdfSphereTrace.
runtime::sdf::SparseSdfDevice HostSdfView(const rt::SdfPrim& sd) {
    runtime::sdf::SparseSdfDevice hdr = sd.header;
    hdr.cell_keys = sd.keys.data();
    hdr.cell_values = sd.values.data();
    hdr.cell_gradients = sd.gradients.data();
    hdr.cell_count = static_cast<uint32_t>(sd.values.size());
    return hdr;
}

bool OracleIntersectLocal(const rt::BlasMesh& mesh, uint32_t local_prim,
                          const math::Vec3& o_l, const math::Vec3& d_l,
                          float t_min, float* out_t) {
    // Declaration-order flat layout: triangles, then spheres, then SDFs.
    if (local_prim < mesh.triangles.size()) {
        const auto& t = mesh.triangles[local_prim];
        float u, v;
        math::Vec3 n;
        return rt::RayTriangleIntersect(o_l, d_l, t.v0, t.v1, t.v2, t_min, out_t,
                                        &u, &v, &n);
    }
    uint32_t s = local_prim - static_cast<uint32_t>(mesh.triangles.size());
    if (s < mesh.spheres.size()) {
        const auto& sp = mesh.spheres[s];
        math::Vec3 n;
        float u, v;
        return rt::RaySphereIntersect(o_l, d_l, sp.center, sp.radius, t_min, out_t,
                                      &n, &u, &v);
    }
    const uint32_t sd_idx = s - static_cast<uint32_t>(mesh.spheres.size());
    if (sd_idx < mesh.sdfs.size()) {
        const auto& sd = mesh.sdfs[sd_idx];
        const auto hdr = HostSdfView(sd);
        math::Vec3 n;
        return rt::RaySdfSphereTrace(o_l, d_l, hdr, sd.aabb, t_min, sd.surface_eps,
                                     sd.max_iters, out_t, &n);
    }
    return false;
}

void OracleClosestHit(const rt::TwoLevelScene& scene, const math::Vec3& o,
                      const math::Vec3& d, float t_min, OracleHit* hit) {
    hit->best_t = rt::RtMissDepth();
    hit->best_prim = rt::kNoPrim;
    for (uint32_t i = 0; i < scene.instances.size(); ++i) {
        const auto& inst = scene.instances[i];
        const auto& mesh = scene.meshes[inst.blas_id];
        const uint32_t prim_count = static_cast<uint32_t>(mesh.triangles.size() +
                                                          mesh.spheres.size() +
                                                          mesh.sdfs.size());
        math::Vec3 o_l, d_l;
        rt::TransformRayToLocal(inst.transform, o, d, &o_l, &d_l);
        for (uint32_t lp = 0; lp < prim_count; ++lp) {
            float t;
            if (OracleIntersectLocal(mesh, lp, o_l, d_l, t_min, &t)) {
                if (t >= t_min) {
                    rt::RtClosestHitUpdate(t, rt::PackPrimId(i, lp), &hit->best_t,
                                           &hit->best_prim);
                }
            }
        }
    }
}

void OracleReconstruct(const rt::TwoLevelScene& scene, uint32_t packed,
                       const math::Vec3& o, const math::Vec3& d,
                       math::Vec3* world_normal, float* u, float* v) {
    uint32_t inst, lp;
    rt::UnpackPrimId(packed, &inst, &lp);
    const auto& I = scene.instances[inst];
    const auto& mesh = scene.meshes[I.blas_id];
    math::Vec3 o_l, d_l;
    rt::TransformRayToLocal(I.transform, o, d, &o_l, &d_l);
    math::Vec3 n_local{0, 0, 0};
    *u = 0.0f;
    *v = 0.0f;
    if (lp < mesh.triangles.size()) {
        const auto& t = mesh.triangles[lp];
        float tt;
        rt::RayTriangleIntersect(o_l, d_l, t.v0, t.v1, t.v2, 0.0f, &tt, u, v, &n_local);
    } else if (lp - mesh.triangles.size() < mesh.spheres.size()) {
        const uint32_t s = lp - static_cast<uint32_t>(mesh.triangles.size());
        const auto& sp = mesh.spheres[s];
        float tt;
        rt::RaySphereIntersect(o_l, d_l, sp.center, sp.radius, 0.0f, &tt, &n_local, u, v);
    } else {
        const uint32_t sd_idx = lp - static_cast<uint32_t>(mesh.triangles.size()) -
                                static_cast<uint32_t>(mesh.spheres.size());
        const auto& sd = mesh.sdfs[sd_idx];
        const auto hdr = HostSdfView(sd);
        float tt;
        rt::RaySdfSphereTrace(o_l, d_l, hdr, sd.aabb, 0.0f, sd.surface_eps,
                              sd.max_iters, &tt, &n_local);
        *u = 0.0f;
        *v = 0.0f;
    }
    *world_normal = rt::QuatRotate(I.transform.rotation, n_local);
}

// Full host two-level render -> a Framebuffer mirroring RenderFrame's AOVs.
rt::Framebuffer OracleRenderTwoLevel(const rt::TwoLevelScene& scene,
                                     const rt::PinholeCamera& cam) {
    rt::Framebuffer fb;
    fb.width = cam.width;
    fb.height = cam.height;
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
            OracleHit hit;
            OracleClosestHit(scene, ray.origin, ray.dir, 0.0f, &hit);
            if (hit.best_prim == rt::kNoPrim) {
                continue;
            }
            const size_t p = static_cast<size_t>(y) * cam.width + x;
            fb.depth[p] = hit.best_t;
            fb.prim[p] = hit.best_prim;

            uint32_t inst, lp;
            rt::UnpackPrimId(hit.best_prim, &inst, &lp);
            const uint32_t mat_id = scene.instances[inst].material_id;

            math::Vec3 n;
            float u, v;
            OracleReconstruct(scene, hit.best_prim, ray.origin, ray.dir, &n, &u, &v);
            fb.normal[p * 3 + 0] = n.x; fb.normal[p * 3 + 1] = n.y; fb.normal[p * 3 + 2] = n.z;
            fb.uv[p * 2 + 0] = u; fb.uv[p * 2 + 1] = v;
            const rt::Material m = scene.materials[mat_id];
            fb.albedo[p * 3 + 0] = m.albedo.x; fb.albedo[p * 3 + 1] = m.albedo.y; fb.albedo[p * 3 + 2] = m.albedo.z;

            const math::Vec3 hitp{
                static_cast<float>(static_cast<double>(ray.origin.x) + static_cast<double>(hit.best_t) * ray.dir.x),
                static_cast<float>(static_cast<double>(ray.origin.y) + static_cast<double>(hit.best_t) * ray.dir.y),
                static_cast<float>(static_cast<double>(ray.origin.z) + static_cast<double>(hit.best_t) * ray.dir.z)};
            const math::Vec3 V = rt::RtNormalize(math::Vec3{-ray.dir.x, -ray.dir.y, -ray.dir.z});
            math::Vec3 L; float light_dist;
            if (scene.light.directional) {
                L = rt::RtNormalize(math::Vec3{-scene.light.direction.x, -scene.light.direction.y, -scene.light.direction.z});
                light_dist = rt::RtMissDepth();
            } else {
                const math::Vec3 to{scene.light.position.x - hitp.x, scene.light.position.y - hitp.y, scene.light.position.z - hitp.z};
                L = rt::RtNormalize(to);
                light_dist = static_cast<float>(std::sqrt(double(to.x)*to.x + double(to.y)*to.y + double(to.z)*to.z));
            }
            math::Vec3 ns = n;
            const double nv = double(n.x)*V.x + double(n.y)*V.y + double(n.z)*V.z;
            if (nv < 0.0) ns = math::Vec3{-n.x, -n.y, -n.z};
            const float seps = 1.0e-3f;
            const math::Vec3 so{hitp.x + ns.x*seps, hitp.y + ns.y*seps, hitp.z + ns.z*seps};
            int lit = 1;
            OracleHit shadow;
            OracleClosestHit(scene, so, L, seps, &shadow);
            if (shadow.best_prim != rt::kNoPrim && shadow.best_t < light_dist - seps) lit = 0;
            const math::Vec3 lc{scene.light.color.x*scene.light.intensity,
                                scene.light.color.y*scene.light.intensity,
                                scene.light.color.z*scene.light.intensity};
            const math::Vec3 c = rt::ShadeDirect(n, V, L, lc, m, lit, scene.ambient.color);
            fb.color[p*3+0]=c.x; fb.color[p*3+1]=c.y; fb.color[p*3+2]=c.z;
        }
    }
    return fb;
}

// ---- Helpers ----

// A quad (2 triangles) at LOCAL z=z0, spanning [-h,h]^2 in x/y. Wound so the
// geometric normal is +Z (cross(e1,e2) = +Z). Mirrors p13's AddQuad.
void AddQuadLocal(rt::BlasMesh& m, float z0, float h) {
    const math::Vec3 a{-h, -h, z0}, b{h, -h, z0}, c{h, h, z0}, d{-h, h, z0};
    m.triangles.push_back({a, b, c, 0u});
    m.triangles.push_back({a, c, d, 0u});
}

// Bake an instance's rigid transform into world-space vertices for the flat p13
// Scene (Oracle 2). Each baked triangle carries the INSTANCE's material_id (the
// two-level model is per-instance; the flat path is per-prim, so we must align).
void BakeInstanceIntoFlat(const rt::TwoLevelScene& scene, uint32_t inst_idx,
                          rt::Scene* flat) {
    const auto& I = scene.instances[inst_idx];
    const auto& mesh = scene.meshes[I.blas_id];
    for (const auto& t : mesh.triangles) {
        rt::TrianglePrim wt;
        wt.v0 = I.transform.TransformPoint(t.v0);
        wt.v1 = I.transform.TransformPoint(t.v1);
        wt.v2 = I.transform.TransformPoint(t.v2);
        wt.material_id = I.material_id;
        flat->triangles.push_back(wt);
    }
    for (const auto& s : mesh.spheres) {
        rt::SpherePrim ws;
        ws.center = I.transform.TransformPoint(s.center);
        ws.radius = s.radius;  // rigid: uniform, no scale
        ws.material_id = I.material_id;
        flat->spheres.push_back(ws);
    }
}

// Cook a unit-sphere narrow-band SDF (p07) in its OWN local frame -> a BlasMesh
// with one SdfPrim. Mirrors test_scene_render.cpp's SDF setup. The aabb is the
// LOCAL bound (the two-level march clips in local space; the TLAS world box folds
// this oriented box by the instance pose).
rt::BlasMesh MakeSphereSdfMesh() {
    using nuka::import::cooker::CookSparseSdf;
    using nuka::import::cooker::SparseSdfParams;
    const auto mesh = nuka::test::SphereMesh(0.0f, 0.0f, 0.0f, 1.0f, 48, 96);
    SparseSdfParams params; params.voxel_size = 0.05f; params.band_voxels = 6u;
    const auto cooked = CookSparseSdf(mesh.vertices.data(),
                                      static_cast<uint32_t>(mesh.vertices.size()/3),
                                      mesh.indices.data(),
                                      static_cast<uint32_t>(mesh.indices.size()/3),
                                      params);
    rt::BlasMesh m;
    rt::SdfPrim sp;
    sp.header.origin = {cooked.origin[0], cooked.origin[1], cooked.origin[2]};
    sp.header.voxel_size = cooked.voxel_size;
    sp.header.dims[0] = cooked.dims[0]; sp.header.dims[1] = cooked.dims[1]; sp.header.dims[2] = cooked.dims[2];
    sp.keys = cooked.cell_keys;
    sp.values = cooked.cell_values;
    sp.gradients.resize(cooked.CellCount());
    for (uint32_t c = 0; c < cooked.CellCount(); ++c)
        sp.gradients[c] = {cooked.cell_gradients[3*c], cooked.cell_gradients[3*c+1], cooked.cell_gradients[3*c+2]};
    sp.aabb.min = {-1.3f, -1.3f, -1.3f};
    sp.aabb.max = {1.3f, 1.3f, 1.3f};
    sp.surface_eps = 0.01f;
    sp.max_iters = 512;
    sp.material_id = 0u;
    m.sdfs.push_back(sp);
    return m;
}

rt::Scene BakeFlat(const rt::TwoLevelScene& scene) {
    rt::Scene flat;
    flat.materials = scene.materials;
    flat.light = scene.light;
    flat.ambient = scene.ambient;
    for (uint32_t i = 0; i < scene.instances.size(); ++i) {
        BakeInstanceIntoFlat(scene, i, &flat);
    }
    return flat;
}

}  // namespace

// =====================================================================
// 0. ORACLE 1 byte-exact host==device for ANY transform (incl. rotation).
//    Mixed scene: two unique meshes, several instances with translation +
//    rotation. This is the D1 + host/device-parity + traversal-independence gate.
// =====================================================================
TEST(RtTwoLevelRender, Oracle1HostDeviceByteExact) {
    rt::TwoLevelScene scene;
    // Mesh 0: a quad (2 triangles, local plane z=5).
    rt::BlasMesh quad;
    AddQuadLocal(quad, 5.0f, 2.0f);
    scene.meshes.push_back(quad);
    // Mesh 1: a single sphere (local center origin, r=1).
    rt::BlasMesh ball;
    ball.spheres.push_back({{0.0f, 0.0f, 0.0f}, 1.0f, 0u});
    scene.meshes.push_back(ball);

    rt::Material m0; m0.albedo = {0.7f, 0.5f, 0.3f}; m0.metallic = 0.0f; m0.roughness = 0.5f;
    rt::Material m1; m1.albedo = {0.2f, 0.6f, 0.9f}; m1.metallic = 0.0f; m1.roughness = 0.6f;
    scene.materials = {m0, m1};
    scene.light.directional = true;
    scene.light.direction = {0.0f, 0.0f, 1.0f};
    scene.ambient.color = {0.04f, 0.04f, 0.04f};

    // Instances: quads at various poses (incl. rotation) + spheres in front.
    auto rotY = math::Quat::FromAxisAngle({0, 1, 0}, 0.4f);
    auto rotX = math::Quat::FromAxisAngle({1, 0, 0}, -0.3f);
    scene.instances.push_back({0u, {{ -1.5f, 0.0f, 0.0f}, rotY}, 0u});
    scene.instances.push_back({0u, {{  1.5f, 0.5f, 1.0f}, rotX}, 0u});
    scene.instances.push_back({1u, {{  0.0f, 0.0f, 3.0f}, math::Quat::Identity()}, 1u});
    scene.instances.push_back({1u, {{ -1.0f, 1.0f, 4.0f}, rotY}, 1u});

    auto cam = rt::BuildPinhole({0.0f, 0.0f, -2.0f}, {0.0f, 0.0f, 1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.6f * kPi, 80u, 64u);

    auto dev = rt::BuildTwoLevelScene(scene);
    auto got = rt::RenderFrame(dev, scene, cam);
    auto ora = OracleRenderTwoLevel(scene, cam);

    size_t hits = 0;
    for (size_t i = 0; i < got.prim.size(); ++i) {
        if (got.prim[i] != rt::kNoPrim) ++hits;
    }
    std::printf("[oracle1] hits=%zu\n", hits);
    EXPECT_GT(hits, 200u);
    // Byte-exact across ALL 6 AOVs (host mirrors the device math VERBATIM).
    EXPECT_EQ(0, std::memcmp(got.depth.data(), ora.depth.data(), got.depth.size()*sizeof(float))) << "depth host!=device";
    EXPECT_EQ(0, std::memcmp(got.prim.data(), ora.prim.data(), got.prim.size()*sizeof(uint32_t))) << "prim host!=device";
    EXPECT_EQ(0, std::memcmp(got.normal.data(), ora.normal.data(), got.normal.size()*sizeof(float))) << "normal host!=device";
    EXPECT_EQ(0, std::memcmp(got.color.data(), ora.color.data(), got.color.size()*sizeof(float))) << "color host!=device";
    EXPECT_EQ(0, std::memcmp(got.albedo.data(), ora.albedo.data(), got.albedo.size()*sizeof(float))) << "albedo host!=device";
    EXPECT_EQ(0, std::memcmp(got.uv.data(), ora.uv.data(), got.uv.size()*sizeof(float))) << "uv host!=device";
}

// =====================================================================
// 1. ORACLE 2 -- IDENTITY instances: byte-exact vs the trusted flat p13
//    RenderScene (proves the two-level nest finds the SAME geometry the
//    independent flat path does). prim_id is NOT compared (different scheme).
// =====================================================================
TEST(RtTwoLevelRender, Oracle2IdentityByteExactVsFlat) {
    rt::TwoLevelScene scene;
    // One quad mesh + one sphere mesh, instanced at IDENTITY at distinct,
    // NON-touching world locations (no coplanar cross-instance ties).
    rt::BlasMesh quad;  AddQuadLocal(quad, 9.0f, 3.0f); scene.meshes.push_back(quad);
    rt::BlasMesh ball;  ball.spheres.push_back({{ -2.0f, 1.5f, 6.0f}, 0.6f, 0u}); scene.meshes.push_back(ball);
    rt::BlasMesh ball2; ball2.spheres.push_back({{ 2.2f, -1.0f, 6.5f}, 0.7f, 0u}); scene.meshes.push_back(ball2);

    rt::Material m0; m0.albedo = {0.7f, 0.5f, 0.3f}; m0.metallic = 0.3f; m0.roughness = 0.4f;
    rt::Material m1; m1.albedo = {0.2f, 0.6f, 0.8f}; m1.metallic = 0.0f; m1.roughness = 0.7f;
    rt::Material m2; m2.albedo = {0.9f, 0.2f, 0.2f}; m2.metallic = 0.0f; m2.roughness = 0.5f;
    scene.materials = {m0, m1, m2};
    scene.light.directional = false;
    scene.light.position = {2.0f, 3.0f, 1.0f};
    scene.light.intensity = 8.0f;
    scene.ambient.color = {0.03f, 0.03f, 0.03f};

    const auto I = math::Quat::Identity();
    scene.instances.push_back({0u, {{0.0f, 0.0f, 0.0f}, I}, 0u}); // quad at identity
    scene.instances.push_back({1u, {{0.0f, 0.0f, 0.0f}, I}, 1u}); // ball 1
    scene.instances.push_back({2u, {{0.0f, 0.0f, 0.0f}, I}, 2u}); // ball 2

    auto cam = rt::BuildPinhole({0.1f, -0.2f, 0.0f}, {0.0f, 0.0f, 6.0f},
                                {0.0f, 1.0f, 0.0f}, 0.8f * kPi, 128u, 96u);

    auto dev = rt::BuildTwoLevelScene(scene);
    auto got = rt::RenderFrame(dev, scene, cam);

    rt::Scene flat = BakeFlat(scene);
    auto flat_fb = rt::RenderScene(flat, cam);

    // IDENTITY instances -> baked world verts are bit-identical to the local
    // verts, so depth/normal/color/albedo memcmp == 0 against the trusted path.
    EXPECT_EQ(0, std::memcmp(got.depth.data(), flat_fb.depth.data(), got.depth.size()*sizeof(float))) << "identity depth != flat";
    EXPECT_EQ(0, std::memcmp(got.normal.data(), flat_fb.normal.data(), got.normal.size()*sizeof(float))) << "identity normal != flat";
    EXPECT_EQ(0, std::memcmp(got.color.data(), flat_fb.color.data(), got.color.size()*sizeof(float))) << "identity color != flat";
    EXPECT_EQ(0, std::memcmp(got.albedo.data(), flat_fb.albedo.data(), got.albedo.size()*sizeof(float))) << "identity albedo != flat";

    size_t hits = 0; for (auto p : got.prim) if (p != rt::kNoPrim) ++hits;
    std::printf("[oracle2-identity] hits=%zu memcmp depth/normal/color/albedo all == 0\n", hits);
    EXPECT_GT(hits, 100u);
}

// =====================================================================
// 1b. ORACLE 2 -- ROTATED/TRANSLATED instances: NEAR-exact vs flat with a
//     STATED tolerance (fp32 QuatInvRotate rounds the local ray differently
//     than pre-rounded flat world vertices -> NOT a byte gate by design).
// =====================================================================
TEST(RtTwoLevelRender, Oracle2RotatedNearExactVsFlat) {
    rt::TwoLevelScene scene;
    rt::BlasMesh quad;  AddQuadLocal(quad, 5.0f, 2.0f); scene.meshes.push_back(quad);

    rt::Material m0; m0.albedo = {0.7f, 0.5f, 0.3f}; m0.metallic = 0.0f; m0.roughness = 0.5f;
    scene.materials = {m0};
    scene.light.directional = true;
    scene.light.direction = {0.0f, 0.0f, 1.0f};
    scene.ambient.color = {0.03f, 0.03f, 0.03f};

    // Two well-separated rotated+translated quads (no cross-instance coplanarity).
    scene.instances.push_back({0u, {{ -1.6f, 0.3f, 0.0f}, math::Quat::FromAxisAngle({0,1,0}, 0.35f)}, 0u});
    scene.instances.push_back({0u, {{  1.8f, -0.4f, 1.5f}, math::Quat::FromAxisAngle({1,0,0}, -0.3f)}, 0u});

    auto cam = rt::BuildPinhole({0.0f, 0.0f, -2.0f}, {0.0f, 0.0f, 1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.6f * kPi, 96u, 72u);

    auto dev = rt::BuildTwoLevelScene(scene);
    auto got = rt::RenderFrame(dev, scene, cam);

    rt::Scene flat = BakeFlat(scene);
    auto flat_fb = rt::RenderScene(flat, cam);

    float max_depth_rel = 0.0f, max_normal_abs = 0.0f;
    size_t common_hits = 0;
    for (size_t i = 0; i < got.depth.size(); ++i) {
        const bool gh = got.prim[i] != rt::kNoPrim;
        const bool fh = flat_fb.prim[i] != rt::kNoPrim;
        if (gh && fh) {
            ++common_hits;
            const float a = got.depth[i], b = flat_fb.depth[i];
            const float rel = std::fabs(a - b) / std::max(1.0f, std::fabs(b));
            max_depth_rel = std::max(max_depth_rel, rel);
            for (int k = 0; k < 3; ++k)
                max_normal_abs = std::max(max_normal_abs, std::fabs(got.normal[3*i+k] - flat_fb.normal[3*i+k]));
        }
    }
    // Stated tolerance: fp32 local-ray rounding differs from pre-rounded flat
    // vertices, so depth is near-exact (rel < 1e-4), not byte-exact.
    const float kDepthRelTol = 1.0e-4f;
    const float kNormalAbsTol = 1.0e-3f;
    std::printf("[oracle2-rotated] common_hits=%zu max_depth_rel=%g (tol %g) max_normal_abs=%g (tol %g)\n",
                common_hits, max_depth_rel, kDepthRelTol, max_normal_abs, kNormalAbsTol);
    EXPECT_GT(common_hits, 100u);
    EXPECT_LT(max_depth_rel, kDepthRelTol);
    EXPECT_LT(max_normal_abs, kNormalAbsTol);
}

// =====================================================================
// 2. ANALYTIC ANCHOR (non-circular): a single triangle inside a ROTATED+
//    TRANSLATED instance. A world ray straight +X from the origin hits it.
//    The quad's local plane z=5 rotates 90 deg about Y to the world plane
//    x = pos.x + 5 = 17, so the +X ray hits at t = 17.0 (rigid: world t ==
//    local t). World hit (17,0,0); world normal (+X). prim_id MUST equal
//    PackPrimId(known_instance, known_local) -- the ONLY check exercising the
//    bit-split + instance-high ordering.
// =====================================================================
TEST(RtTwoLevelRender, AnalyticAnchorRotatedTranslatedTriangleAndPrimId) {
    rt::TwoLevelScene scene;
    // Mesh 0: a dummy small quad far away (so the anchor instance is index 1,
    // exercising a NON-zero instance id in the packed prim_id high bits).
    rt::BlasMesh dummy; AddQuadLocal(dummy, 5.0f, 0.2f); scene.meshes.push_back(dummy);
    // Mesh 1: the big anchor quad (local plane z=5, half-extent 4).
    rt::BlasMesh anchor; AddQuadLocal(anchor, 5.0f, 4.0f); scene.meshes.push_back(anchor);

    rt::Material m0; m0.albedo = {0.3f, 0.3f, 0.3f};
    rt::Material m1; m1.albedo = {0.8f, 0.6f, 0.2f}; m1.metallic = 0.0f; m1.roughness = 0.5f;
    scene.materials = {m0, m1};
    scene.light.directional = true;
    scene.light.direction = {-1.0f, 0.0f, 0.0f};  // L (to light) = +X, toward camera-side
    scene.ambient.color = {0.05f, 0.05f, 0.05f};

    // Instance 0: dummy parked far off the +X ray (won't be hit by the center ray).
    scene.instances.push_back({0u, {{ -50.0f, -50.0f, -50.0f}, math::Quat::Identity()}, 0u});
    // Instance 1 (the anchor): rotate 90 deg about Y, translate (12,1,2).
    const float ang = kPi * 0.5f;
    scene.instances.push_back({1u, {{ 12.0f, 1.0f, 2.0f}, math::Quat::FromAxisAngle({0,1,0}, ang)}, 1u});

    // Camera at origin looking +X; the dead-center pixel of an ODD-res image
    // fires exactly the (1,0,0) world ray.
    auto cam = rt::BuildPinhole({0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
                                {0.0f, 1.0f, 0.0f}, 0.3f * kPi, 65u, 65u);

    auto dev = rt::BuildTwoLevelScene(scene);
    auto got = rt::RenderFrame(dev, scene, cam);

    const uint32_t cx = 32, cy = 32;  // dead-center (odd res)
    const size_t cp = static_cast<size_t>(cy) * cam.width + cx;
    ASSERT_NE(got.prim[cp], rt::kNoPrim) << "center +X ray missed the anchor";

    // Hand-computed world depth: local plane z=5 -> world x=17 -> t=17.0.
    std::printf("[anchor] depth=%.9g (expect 17.0) prim=0x%08x\n", got.depth[cp], got.prim[cp]);
    EXPECT_NEAR(got.depth[cp], 17.0f, 1e-3f);

    // World normal: local +Z rotated 90 deg about Y -> world +X.
    EXPECT_NEAR(got.normal[3*cp+0], 1.0f, 1e-4f);
    EXPECT_NEAR(got.normal[3*cp+1], 0.0f, 1e-4f);
    EXPECT_NEAR(std::fabs(got.normal[3*cp+2]), 0.0f, 1e-4f);

    // World hit point = origin + t*(+X) = (17,0,0). Check via depth+ray (center
    // ray is +X), so hit.x == depth, hit.y == hit.z == 0.
    // (depth already asserted; y/z are 0 by construction of the +X ray.)

    // THE packing gate: instance 1, local triangle 0 (the +X ray hits the FIRST
    // triangle of the quad, a=(-4,-4,5),b=(4,-4,5),c=(4,4,5), which contains the
    // local hit (0,0,5)). This is the ONLY check exercising PackPrimId's bit-split
    // + instance-HIGH ordering.
    const uint32_t expect_packed = rt::PackPrimId(1u, 0u);
    EXPECT_EQ(got.prim[cp], expect_packed)
        << "packed prim_id wrong: got 0x" << std::hex << got.prim[cp]
        << " expect 0x" << expect_packed << " (PackPrimId(1,0))";
    // Sanity: unpack round-trips.
    uint32_t inst, local;
    rt::UnpackPrimId(got.prim[cp], &inst, &local);
    EXPECT_EQ(inst, 1u);
    EXPECT_EQ(local, 0u);
}

// =====================================================================
// 2b. p13's hand-computed single-triangle anchor through the two-level path
//     at IDENTITY: t == 5.0497524691810, u==v==0.25, normal +/-Z. Confirms the
//     nest reproduces the trusted p13 ground truth.
// =====================================================================
TEST(RtTwoLevelRender, AnalyticAnchorP13TriangleIdentity) {
    rt::TwoLevelScene scene;
    rt::BlasMesh m;
    m.triangles.push_back({{0,0,5}, {2,0,5}, {0,2,5}, 0u});
    scene.meshes.push_back(m);
    scene.materials = {rt::Material{}};
    scene.light.directional = true;
    scene.light.direction = {0,0,1};
    scene.instances.push_back({0u, math::Transform::Identity(), 0u});

    // Camera at origin; aim the center ray at (0.5,0.5,5) like p13's anchor by
    // sampling that pixel via the oracle (here we just check the center hit t).
    auto cam = rt::BuildPinhole({0,0,0}, {0.5f, 0.5f, 5.0f}, {0,1,0}, 0.2f*kPi, 65u, 65u);
    auto dev = rt::BuildTwoLevelScene(scene);
    auto got = rt::RenderFrame(dev, scene, cam);

    const size_t cp = static_cast<size_t>(32)*cam.width + 32;
    ASSERT_NE(got.prim[cp], rt::kNoPrim);
    // Center ray points at the triangle interior; t close to the analytic 5.0497.
    std::printf("[anchor-p13] center depth=%.10g (analytic ~5.0497524691810)\n", got.depth[cp]);
    EXPECT_NEAR(got.depth[cp], 5.0497524691810f, 1e-3f);
    EXPECT_EQ(got.prim[cp], rt::PackPrimId(0u, 0u));
}

// =====================================================================
// 3. INSTANCE-TRANSFORM TRACKING: an anchored background pixel becomes the
//    instance's color after the instance moves onto the +X ray. The BLAS is
//    reused (built once); only the per-frame TLAS changes.
// =====================================================================
TEST(RtTwoLevelRender, InstanceTransformTracking) {
    rt::TwoLevelScene scene;
    rt::BlasMesh quad; AddQuadLocal(quad, 5.0f, 4.0f); scene.meshes.push_back(quad);
    rt::Material m1; m1.albedo = {0.9f, 0.1f, 0.1f}; m1.metallic = 0.0f; m1.roughness = 0.5f;
    scene.materials = {m1};
    scene.light.directional = true;
    scene.light.direction = {-1.0f, 0.0f, 0.0f};
    scene.ambient.color = {0.05f, 0.05f, 0.05f};

    // Start: the quad is parked far off the +X center ray (background at center).
    scene.instances.push_back({0u, {{ 12.0f, 100.0f, 0.0f}, math::Quat::FromAxisAngle({0,1,0}, kPi*0.5f)}, 0u});

    auto cam = rt::BuildPinhole({0,0,0}, {1.0f,0,0}, {0,1,0}, 0.3f*kPi, 65u, 65u);

    auto dev = rt::BuildTwoLevelScene(scene);
    auto before = rt::RenderFrame(dev, scene, cam);
    const size_t cp = static_cast<size_t>(32)*cam.width + 32;
    ASSERT_EQ(before.prim[cp], rt::kNoPrim) << "center should be background before move";

    // Move the instance onto the +X ray (same BLAS, new TLAS).
    scene.instances[0].transform.position = {12.0f, 1.0f, 2.0f};
    auto after = rt::RenderFrame(dev, scene, cam);
    ASSERT_NE(after.prim[cp], rt::kNoPrim) << "center should hit the moved instance";
    EXPECT_EQ(after.prim[cp], rt::PackPrimId(0u, 0u));
    // The center pixel now carries the instance's (red) albedo.
    std::printf("[tracking] after albedo=(%g,%g,%g)\n",
                after.albedo[3*cp+0], after.albedo[3*cp+1], after.albedo[3*cp+2]);
    EXPECT_NEAR(after.albedo[3*cp+0], 0.9f, 1e-5f);
    EXPECT_NEAR(after.albedo[3*cp+1], 0.1f, 1e-5f);
    EXPECT_NEAR(after.albedo[3*cp+2], 0.1f, 1e-5f);
}

// =====================================================================
// 4. D1: two-run byte-exact across ALL 6 AOVs (mirror p13's gate).
// =====================================================================
TEST(RtTwoLevelRender, D1TwoRunByteExactAllAovs) {
    rt::TwoLevelScene scene;
    rt::BlasMesh quad; AddQuadLocal(quad, 9.0f, 5.0f); scene.meshes.push_back(quad);
    rt::BlasMesh ball; ball.spheres.push_back({{0,0,0}, 0.5f, 0u}); scene.meshes.push_back(ball);

    rt::Material m0; m0.albedo = {0.7f, 0.5f, 0.3f}; m0.metallic = 0.3f; m0.roughness = 0.4f;
    rt::Material m1; m1.albedo = {0.2f, 0.6f, 0.8f}; m1.metallic = 0.0f; m1.roughness = 0.7f;
    scene.materials = {m0, m1};
    scene.light.directional = false;
    scene.light.position = {2.0f, 3.0f, 1.0f};
    scene.light.intensity = 8.0f;

    // One big quad backdrop + a 5x5 grid of sphere instances (reused BLAS).
    scene.instances.push_back({0u, math::Transform::Identity(), 0u});
    for (int gx = -2; gx <= 2; ++gx)
        for (int gy = -2; gy <= 2; ++gy) {
            math::Transform t; t.position = {gx*1.4f, gy*1.4f, 6.0f};
            scene.instances.push_back({1u, t, 1u});
        }

    auto cam = rt::BuildPinhole({0.1f, -0.2f, 0.0f}, {0.0f, 0.0f, 6.0f},
                                {0.0f, 1.0f, 0.0f}, 0.8f * kPi, 128u, 96u);

    auto dev = rt::BuildTwoLevelScene(scene);
    auto r1 = rt::RenderFrame(dev, scene, cam);
    auto r2 = rt::RenderFrame(dev, scene, cam);
    ASSERT_EQ(r1.color.size(), r2.color.size());
    EXPECT_EQ(0, std::memcmp(r1.color.data(), r2.color.data(), r1.color.size()*sizeof(float))) << "color not byte-exact";
    EXPECT_EQ(0, std::memcmp(r1.depth.data(), r2.depth.data(), r1.depth.size()*sizeof(float))) << "depth not byte-exact";
    EXPECT_EQ(0, std::memcmp(r1.normal.data(), r2.normal.data(), r1.normal.size()*sizeof(float))) << "normal not byte-exact";
    EXPECT_EQ(0, std::memcmp(r1.albedo.data(), r2.albedo.data(), r1.albedo.size()*sizeof(float))) << "albedo not byte-exact";
    EXPECT_EQ(0, std::memcmp(r1.uv.data(), r2.uv.data(), r1.uv.size()*sizeof(float))) << "uv not byte-exact";
    EXPECT_EQ(0, std::memcmp(r1.prim.data(), r2.prim.data(), r1.prim.size()*sizeof(uint32_t))) << "prim not byte-exact";
    std::printf("[d1] all 6 AOVs two-run byte-exact (%ux%u, %zu instances)\n",
                cam.width, cam.height, scene.instances.size());
}

// =====================================================================
// 5. SPARSE-SDF leaf through the two-level nest (the p16-relevant path). Two
//    sphere-SDF instances: ONE at IDENTITY, ONE ROTATED. Gates:
//      * Oracle 1 byte-exact host==device across ALL 6 AOVs (the rotated SDF
//        exercises ReconstructHit's QuatRotate(rotation, n_local) gradient-
//        normal-to-world line + the local-AABB march clip under rotation), and
//      * Oracle 2 byte-exact vs the trusted flat p13 RenderScene for the IDENTITY
//        SDF (SDF renders at identity in the flat path; QuatRotate(identity)==id,
//        so the two-level identity SDF must be bit-identical to flat).
// =====================================================================
TEST(RtTwoLevelRender, SdfLeafThroughNestIdentityAndRotated) {
    rt::BlasMesh sdf_mesh = MakeSphereSdfMesh();

    // --- Sub-case A: IDENTITY SDF instance, byte-exact vs flat (Oracle 2). ---
    {
        rt::TwoLevelScene scene;
        scene.meshes.push_back(sdf_mesh);
        scene.materials = {rt::Material{}};
        scene.light.directional = true;
        scene.light.direction = {0.0f, 0.0f, 1.0f};
        scene.instances.push_back({0u, math::Transform::Identity(), 0u});

        auto cam = rt::BuildPinhole({0.0f, 0.0f, -4.0f}, {0.0f, 0.0f, 0.0f},
                                    {0.0f, 1.0f, 0.0f}, 0.5f * kPi, 64u, 64u);
        auto dev = rt::BuildTwoLevelScene(scene);
        auto got = rt::RenderFrame(dev, scene, cam);
        auto ora = OracleRenderTwoLevel(scene, cam);

        size_t hits = 0; for (auto p : got.prim) if (p != rt::kNoPrim) ++hits;
        std::printf("[sdf-identity] hits=%zu\n", hits);
        EXPECT_GT(hits, 100u);
        // Oracle 1: host==device byte-exact (shared RaySdfSphereTrace).
        EXPECT_EQ(0, std::memcmp(got.depth.data(), ora.depth.data(), got.depth.size()*sizeof(float))) << "sdf depth host!=device";
        EXPECT_EQ(0, std::memcmp(got.prim.data(), ora.prim.data(), got.prim.size()*sizeof(uint32_t))) << "sdf prim host!=device";
        EXPECT_EQ(0, std::memcmp(got.normal.data(), ora.normal.data(), got.normal.size()*sizeof(float))) << "sdf normal host!=device";

        // Oracle 2: identity SDF instance == the trusted flat path (one SdfPrim).
        rt::Scene flat;
        flat.sdfs.push_back(sdf_mesh.sdfs[0]);
        flat.materials = scene.materials;
        flat.light = scene.light;
        flat.ambient = scene.ambient;
        auto flat_fb = rt::RenderScene(flat, cam);
        EXPECT_EQ(0, std::memcmp(got.depth.data(), flat_fb.depth.data(), got.depth.size()*sizeof(float))) << "sdf identity depth != flat";
        EXPECT_EQ(0, std::memcmp(got.normal.data(), flat_fb.normal.data(), got.normal.size()*sizeof(float))) << "sdf identity normal != flat";
        std::printf("[sdf-identity] Oracle1 + Oracle2-vs-flat memcmp == 0\n");

        // Analytic: center ray from z=-4 hits the unit sphere front at z=-1 ->
        // depth 3.0 within (surface_eps + voxel).
        const size_t cp = static_cast<size_t>(32)*cam.width + 32;
        ASSERT_NE(got.prim[cp], rt::kNoPrim);
        EXPECT_LT(std::fabs(got.depth[cp] - 3.0f), sdf_mesh.sdfs[0].surface_eps + 0.05f);
    }

    // --- Sub-case B: ROTATED SDF instance, Oracle 1 byte-exact (exercises the
    //     gradient-normal rotation + local-AABB march clip under rotation). ---
    {
        rt::TwoLevelScene scene;
        scene.meshes.push_back(sdf_mesh);
        scene.materials = {rt::Material{}};
        scene.light.directional = true;
        scene.light.direction = {0.0f, 0.0f, 1.0f};
        // Rotate about a tilted axis + translate (the sphere SDF is symmetric, but
        // the gradient-normal-to-world rotation + local-frame ray transform are NOT
        // trivial under this pose -> a real exercise of the rotated SDF path).
        const auto rot = math::Quat::FromAxisAngle({0.3f, 1.0f, 0.2f}, 0.7f);
        scene.instances.push_back({0u, {{0.4f, -0.3f, 0.5f}, rot}, 0u});

        auto cam = rt::BuildPinhole({0.0f, 0.0f, -4.0f}, {0.0f, 0.0f, 0.0f},
                                    {0.0f, 1.0f, 0.0f}, 0.5f * kPi, 64u, 64u);
        auto dev = rt::BuildTwoLevelScene(scene);
        auto got = rt::RenderFrame(dev, scene, cam);
        auto ora = OracleRenderTwoLevel(scene, cam);

        size_t hits = 0; for (auto p : got.prim) if (p != rt::kNoPrim) ++hits;
        std::printf("[sdf-rotated] hits=%zu\n", hits);
        EXPECT_GT(hits, 100u);
        EXPECT_EQ(0, std::memcmp(got.depth.data(), ora.depth.data(), got.depth.size()*sizeof(float))) << "rotated sdf depth host!=device";
        EXPECT_EQ(0, std::memcmp(got.prim.data(), ora.prim.data(), got.prim.size()*sizeof(uint32_t))) << "rotated sdf prim host!=device";
        EXPECT_EQ(0, std::memcmp(got.normal.data(), ora.normal.data(), got.normal.size()*sizeof(float))) << "rotated sdf normal host!=device (gradient-rotate line)";
        EXPECT_EQ(0, std::memcmp(got.color.data(), ora.color.data(), got.color.size()*sizeof(float))) << "rotated sdf color host!=device";
        std::printf("[sdf-rotated] Oracle1 byte-exact across depth/prim/normal/color\n");
    }
}
