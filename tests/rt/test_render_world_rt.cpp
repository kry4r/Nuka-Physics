// ---------------------------------------------------------------------------
// M11 RT-5: the offscreen D1 gate for the RenderWorld -> rt_adapter -> RtBackendI
// path (§3.10 RT row). Proves the engine-layer backend-agnostic RT interface
// drives the self-written two-level tracer to an image:
//
//   (a) D1: two RtBackendI dispatches of the adapter-produced scene are 6-AOV
//       byte-identical (memcmp == 0) -- the relocated kernels are still
//       deterministic through the interface.
//   (b) PARITY: the adapter-produced TwoLevelScene renders 6-AOV byte-identical
//       to a HAND-BUILT TwoLevelScene oracle carrying the SAME geometry /
//       transforms / materials / light -- the adapter mapping is faithful.
//   (c) COVERAGE: the rendered frame has > 0 hit pixels (a non-trivial image).
//
// Built -ffp-contract=off to pair with the kernel's --fmad=false (host/device
// bit-exactness is irrelevant here since we compare device-vs-device, but the
// flag keeps this TU consistent with the other rt tests). In the CUDA region
// OUTSIDE NK_BUILD_VULKAN (this box runs CUDA RT to an image buffer locally).
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "render/render_world.hpp"
#include "render/rt_adapter.hpp"
#include "render/rt_backend.hpp"
#include "phi/backend_cuda/rt/ray_box.cuh"  // host-callable kNoPrim sentinel
#include "rt/camera.hpp"
#include "rt/framebuffer.hpp"
#include "rt/two_level_render.hpp"
#include "scene/ecs/components.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace nuka;

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr uint32_t kW = 128u;
constexpr uint32_t kH = 96u;

// A unit box (half-extents) as positions + indices (12 triangles). Matches the
// face winding the in-tree MakeBoxMesh uses so geometry is comparable.
render::MeshGeometry BoxGeometry(math::Vec3 he) {
    const float x = he.x, y = he.y, z = he.z;
    render::MeshGeometry g;
    g.positions = {
        -x, -y, -z,  x, -y, -z,  x,  y, -z, -x,  y, -z,
        -x, -y,  z,  x, -y,  z,  x,  y,  z, -x,  y,  z,
    };
    // +Z, -Z, +X, -X, +Y, -Y (two tris each), same as MakeBoxMesh.
    g.indices = {
        4,5,6, 4,6,7,   0,3,2, 0,2,1,   1,2,6, 1,6,5,
        0,4,7, 0,7,3,   2,3,7, 2,7,6,   0,1,5, 0,5,4,
    };
    return g;
}

// rt-side equivalent of BoxGeometry, for the hand-built oracle.
rt::BlasMesh BoxBlas(math::Vec3 he) {
    const render::MeshGeometry g = BoxGeometry(he);
    rt::BlasMesh blas;
    for (size_t t = 0; t < g.indices.size(); t += 3) {
        const auto v = [&g](uint32_t i) -> math::Vec3 {
            return {g.positions[i * 3 + 0], g.positions[i * 3 + 1], g.positions[i * 3 + 2]};
        };
        blas.triangles.push_back({v(g.indices[t]), v(g.indices[t + 1]), v(g.indices[t + 2]), 0u});
    }
    return blas;
}

scene::RenderMaterial MakeMat(float r, float g, float b, float metal, float rough) {
    scene::RenderMaterial m;
    m.base_color[0] = r; m.base_color[1] = g; m.base_color[2] = b; m.base_color[3] = 1.0f;
    m.metallic = metal;
    m.roughness = rough;
    return m;
}

// Build a minimal RenderWorld: a floor box + a small box, two materials, a camera
// and a point light. Hand-constructed (no Registry/cook) so the test is local and
// hermetic.
render::RenderWorld BuildTestRenderWorld() {
    render::RenderWorld world;

    const uint32_t floor_mesh =
        world.meshes.InternPrimitive("floor", [] { return BoxGeometry({2.5f, 2.5f, 0.05f}); });
    const uint32_t cube_mesh =
        world.meshes.InternPrimitive("cube", [] { return BoxGeometry({0.4f, 0.4f, 0.4f}); });

    world.materials.push_back(MakeMat(0.10f, 0.11f, 0.13f, 0.0f, 0.70f)); // floor
    world.materials.push_back(MakeMat(0.80f, 0.30f, 0.25f, 0.10f, 0.40f)); // cube

    render::RenderInstance floor_inst;
    floor_inst.mesh_id = floor_mesh;
    floor_inst.render_material_id = 0u;
    floor_inst.world_xform = {{0.0f, 0.0f, -0.05f}, math::Quat::Identity()};
    world.instances.push_back(floor_inst);

    render::RenderInstance cube_inst;
    cube_inst.mesh_id = cube_mesh;
    cube_inst.render_material_id = 1u;
    cube_inst.world_xform = {{0.0f, 0.0f, 0.45f},
                             math::Quat::FromAxisAngle({0.0f, 0.0f, 1.0f}, 0.4f)};
    world.instances.push_back(cube_inst);

    render::RenderLight light;
    light.color = {1.0f, 0.95f, 0.9f};
    light.intensity = 6.0f;
    light.world_xform = {{-2.0f, -2.5f, 3.0f}, math::Quat::Identity()};
    world.lights.push_back(light);

    return world;
}

// The hand-built TwoLevelScene oracle: SAME geometry/transforms/materials/light
// as BuildTestRenderWorld, expressed directly in rt types (the trusted target
// the adapter must reproduce). The adapter appends a default material at the end,
// so the oracle mirrors that (materials[0]=floor, [1]=cube, [2]=default).
rt::TwoLevelScene BuildOracleScene() {
    rt::TwoLevelScene s;
    s.meshes.push_back(BoxBlas({2.5f, 2.5f, 0.05f})); // floor (mesh 0)
    s.meshes.push_back(BoxBlas({0.4f, 0.4f, 0.4f}));  // cube  (mesh 1)

    rt::Material floor_mat; floor_mat.albedo = {0.10f, 0.11f, 0.13f}; floor_mat.metallic = 0.0f; floor_mat.roughness = 0.70f;
    rt::Material cube_mat;  cube_mat.albedo  = {0.80f, 0.30f, 0.25f}; cube_mat.metallic  = 0.10f; cube_mat.roughness  = 0.40f;
    s.materials = {floor_mat, cube_mat, rt::Material{} /*default*/};

    s.instances.push_back({0u, {{0.0f, 0.0f, -0.05f}, math::Quat::Identity()}, 0u});
    s.instances.push_back({1u, {{0.0f, 0.0f, 0.45f},
                                 math::Quat::FromAxisAngle({0.0f, 0.0f, 1.0f}, 0.4f)}, 1u});

    s.light.directional = false;
    s.light.position = {-2.0f, -2.5f, 3.0f};
    s.light.color = {1.0f, 0.95f, 0.9f};
    s.light.intensity = 6.0f;
    return s;
}

rt::PinholeCamera TestCamera() {
    return rt::BuildPinhole({-1.6f, -2.0f, 1.4f}, {0.0f, 0.0f, 0.4f},
                            {0.0f, 0.0f, 1.0f}, 0.35f * kPi, kW, kH);
}

// memcmp all 6 AOVs of two framebuffers; returns true iff byte-identical.
::testing::AssertionResult FramebuffersByteEqual(const rt::Framebuffer& a,
                                                 const rt::Framebuffer& b) {
    if (a.width != b.width || a.height != b.height) {
        return ::testing::AssertionFailure() << "dimension mismatch";
    }
    const auto cmp = [](const auto& x, const auto& y, const char* name) -> ::testing::AssertionResult {
        if (x.size() != y.size()) {
            return ::testing::AssertionFailure() << name << " size mismatch";
        }
        if (!x.empty() && std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])) != 0) {
            return ::testing::AssertionFailure() << name << " bytes differ";
        }
        return ::testing::AssertionSuccess();
    };
    auto r = cmp(a.color, b.color, "color");   if (!r) return r;
    r = cmp(a.depth, b.depth, "depth");        if (!r) return r;
    r = cmp(a.normal, b.normal, "normal");     if (!r) return r;
    r = cmp(a.albedo, b.albedo, "albedo");     if (!r) return r;
    r = cmp(a.uv, b.uv, "uv");                 if (!r) return r;
    r = cmp(a.prim, b.prim, "prim");           if (!r) return r;
    return ::testing::AssertionSuccess();
}

size_t HitPixels(const rt::Framebuffer& fb) {
    size_t hits = 0;
    for (uint32_t p : fb.prim) {
        if (p != rt::kNoPrim) ++hits;
    }
    return hits;
}

}  // namespace

// (a) D1: two RtBackendI dispatches of the adapter scene are 6-AOV byte-identical.
TEST(RenderWorldRt, AdapterDispatchTwoRunByteExactAllAovs) {
    auto backend = render::CreateCudaRtBackend();
    ASSERT_NE(backend, nullptr) << "no CUDA RT backend available";

    const render::RenderWorld world = BuildTestRenderWorld();
    const rt::TwoLevelScene scene = render::RenderWorldToTwoLevelScene(world);
    const rt::PinholeCamera camera = TestCamera();

    render::RtSceneHandle* h = backend->BuildScene(scene);
    ASSERT_NE(h, nullptr);
    const rt::Framebuffer run1 = backend->TraceToHost(h, scene, camera);
    const rt::Framebuffer run2 = backend->TraceToHost(h, scene, camera);
    backend->FreeScene(h);

    EXPECT_TRUE(FramebuffersByteEqual(run1, run2));
}

// (b) PARITY: adapter scene renders byte-identical to the hand-built oracle.
TEST(RenderWorldRt, AdapterMatchesHandBuiltOracleByteExact) {
    auto backend = render::CreateCudaRtBackend();
    ASSERT_NE(backend, nullptr) << "no CUDA RT backend available";

    const render::RenderWorld world = BuildTestRenderWorld();
    const rt::TwoLevelScene adapter_scene = render::RenderWorldToTwoLevelScene(world);
    const rt::TwoLevelScene oracle_scene = BuildOracleScene();
    const rt::PinholeCamera camera = TestCamera();

    render::RtSceneHandle* ha = backend->BuildScene(adapter_scene);
    render::RtSceneHandle* ho = backend->BuildScene(oracle_scene);
    ASSERT_NE(ha, nullptr);
    ASSERT_NE(ho, nullptr);
    const rt::Framebuffer fa = backend->TraceToHost(ha, adapter_scene, camera);
    const rt::Framebuffer fo = backend->TraceToHost(ho, oracle_scene, camera);
    backend->FreeScene(ha);
    backend->FreeScene(ho);

    EXPECT_TRUE(FramebuffersByteEqual(fa, fo));
}

// (b') BufferI output path: Trace into caller-provided phi v2 device buffers and
// download equals the host-convenience TraceToHost (the OD-12 output contract).
TEST(RenderWorldRt, BufferIOutputMatchesHostConvenience) {
    auto backend = render::CreateCudaRtBackend();
    ASSERT_NE(backend, nullptr) << "no CUDA RT backend available";

    const render::RenderWorld world = BuildTestRenderWorld();
    const rt::TwoLevelScene scene = render::RenderWorldToTwoLevelScene(world);
    const rt::PinholeCamera camera = TestCamera();

    render::RtSceneHandle* h = backend->BuildScene(scene);
    ASSERT_NE(h, nullptr);

    const rt::Framebuffer host = backend->TraceToHost(h, scene, camera);

    // Allocate the 6 AOV device buffers from the backend's AOV BufferType.
    phi::BufferType* bt = backend->AovBufferType();
    ASSERT_NE(bt, nullptr);
    const size_t px = static_cast<size_t>(kW) * static_cast<size_t>(kH);
    render::RtAovBuffers aov;
    aov.color  = phi::BufferAlloc(bt, px * 3u * sizeof(float));
    aov.depth  = phi::BufferAlloc(bt, px * sizeof(float));
    aov.normal = phi::BufferAlloc(bt, px * 3u * sizeof(float));
    aov.albedo = phi::BufferAlloc(bt, px * 3u * sizeof(float));
    aov.uv     = phi::BufferAlloc(bt, px * 2u * sizeof(float));
    aov.prim   = phi::BufferAlloc(bt, px * sizeof(uint32_t));

    backend->Trace(h, scene, camera, aov);

    // Download and compare to the host-convenience frame.
    rt::Framebuffer dev;
    dev.width = kW; dev.height = kH;
    dev.color.resize(px * 3u);
    dev.depth.resize(px);
    dev.normal.resize(px * 3u);
    dev.albedo.resize(px * 3u);
    dev.uv.resize(px * 2u);
    dev.prim.resize(px);
    phi::BufferDownload(aov.color,  dev.color.data(),  0, dev.color.size()  * sizeof(float));
    phi::BufferDownload(aov.depth,  dev.depth.data(),  0, dev.depth.size()  * sizeof(float));
    phi::BufferDownload(aov.normal, dev.normal.data(), 0, dev.normal.size() * sizeof(float));
    phi::BufferDownload(aov.albedo, dev.albedo.data(), 0, dev.albedo.size() * sizeof(float));
    phi::BufferDownload(aov.uv,     dev.uv.data(),     0, dev.uv.size()     * sizeof(float));
    phi::BufferDownload(aov.prim,   dev.prim.data(),   0, dev.prim.size()   * sizeof(uint32_t));

    phi::BufferFree(aov.color);
    phi::BufferFree(aov.depth);
    phi::BufferFree(aov.normal);
    phi::BufferFree(aov.albedo);
    phi::BufferFree(aov.uv);
    phi::BufferFree(aov.prim);
    backend->FreeScene(h);

    EXPECT_TRUE(FramebuffersByteEqual(host, dev));
}

// (c) COVERAGE: the rendered frame is non-trivial (> 0 hit pixels).
TEST(RenderWorldRt, NonTrivialCoverage) {
    auto backend = render::CreateCudaRtBackend();
    ASSERT_NE(backend, nullptr) << "no CUDA RT backend available";

    const render::RenderWorld world = BuildTestRenderWorld();
    const rt::TwoLevelScene scene = render::RenderWorldToTwoLevelScene(world);
    const rt::PinholeCamera camera = TestCamera();

    render::RtSceneHandle* h = backend->BuildScene(scene);
    ASSERT_NE(h, nullptr);
    const rt::Framebuffer fb = backend->TraceToHost(h, scene, camera);
    backend->FreeScene(h);

    EXPECT_EQ(fb.width, kW);
    EXPECT_EQ(fb.height, kH);
    EXPECT_GT(HitPixels(fb), 0u) << "rendered frame has no geometry";
}
