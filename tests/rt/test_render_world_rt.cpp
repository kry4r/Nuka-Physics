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
#include "render/mesh_normals.hpp"
#include "render/render_world.hpp"
#include "render/rt_adapter.hpp"
#include "render/rt_backend.hpp"
#include "phi/backend_cuda/rt/ray_box.cuh"  // host-callable kNoPrim sentinel
#include "rt/camera.hpp"
#include "rt/framebuffer.hpp"
#include "rt/two_level_render.hpp"
#include "scene/ecs/components.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
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

// A coarse UV sphere as positions + indices -- faceted enough that smooth
// per-vertex normals visibly diverge from the flat per-face fallback.
render::MeshGeometry SphereGeometry(float radius, uint32_t rings, uint32_t sectors) {
    render::MeshGeometry g;
    for (uint32_t r = 0; r <= rings; ++r) {
        const float theta = static_cast<float>(r) / static_cast<float>(rings) * kPi;
        for (uint32_t s = 0; s <= sectors; ++s) {
            const float phi = static_cast<float>(s) / static_cast<float>(sectors) * 2.0f * kPi;
            const float x = std::sin(theta) * std::cos(phi);
            const float y = std::cos(theta);
            const float z = std::sin(theta) * std::sin(phi);
            g.positions.push_back(radius * x);
            g.positions.push_back(radius * y);
            g.positions.push_back(radius * z);
        }
    }
    const uint32_t stride = sectors + 1u;
    for (uint32_t r = 0; r < rings; ++r) {
        for (uint32_t s = 0; s < sectors; ++s) {
            const uint32_t a = r * stride + s, b = a + stride;
            g.indices.insert(g.indices.end(), {a, a + 1u, b, b, a + 1u, b + 1u});
        }
    }
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

// ---------------------------------------------------------------------------
// Smooth-normal + refractive-dielectric beauty coverage: an opaque transmission==0
// no-op gate, a transmissive slab that bends + Beer-darkens the background, and a
// transmissive sphere whose refraction shifts when per-vertex normals are supplied.
// ---------------------------------------------------------------------------

constexpr uint32_t kRW = 200u;
constexpr uint32_t kRH = 200u;

// Order-independent FNV-1a checksum over a float buffer's raw bytes (the no-op
// gate prints this so a clean-tree run can be compared bit-for-bit).
uint64_t Fnv1a(const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    const auto* p = reinterpret_cast<const unsigned char*>(v.data());
    const size_t n = v.size() * sizeof(float);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

void WritePpm(const rt::Framebuffer& fb, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%u %u\n255\n", fb.width, fb.height);
    std::vector<unsigned char> row(static_cast<size_t>(fb.width) * 3u);
    for (uint32_t y = 0; y < fb.height; ++y) {
        for (uint32_t x = 0; x < fb.width; ++x) {
            const size_t i = (static_cast<size_t>(y) * fb.width + x) * 3u;
            for (int c = 0; c < 3; ++c) {
                float v = fb.color[i + c];
                v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                row[x * 3u + c] = static_cast<unsigned char>(v * 255.0f + 0.5f);
            }
        }
        std::fwrite(row.data(), 1, row.size(), f);
    }
    std::fclose(f);
}

rt::BeautyOptions FixedBeauty() {
    rt::BeautyOptions o;
    o.samples = 8u;
    o.shadow_rays = 2u;
    o.ao_samples = 2u;
    o.gi_bounces = 1u;
    o.seed = 0x1234abcdu;
    o.transmit_bounces = 2u;
    return o;
}

// Camera looking along +Y at a back wall, with room for a slab in between.
rt::PinholeCamera RefractCamera() {
    return rt::BuildPinhole({0.0f, -3.0f, 0.6f}, {0.0f, 1.5f, 0.6f},
                            {0.0f, 0.0f, 1.0f}, 0.40f * kPi, kRW, kRH);
}

// A back wall with a bright vertical bar (the background "feature" refraction
// bends), plus an optional transmissive slab in front. transmission/absorption/
// slab_hy parameterize the dielectric; transmission<=0 => an opaque control slab.
render::RenderWorld BuildRefractWorld(bool with_slab, float transmission, float absorption,
                                      float slab_hy) {
    render::RenderWorld world;
    const uint32_t wall = world.meshes.InternPrimitive(
        "wall", [] { return BoxGeometry({2.5f, 0.05f, 1.6f}); });
    const uint32_t bar = world.meshes.InternPrimitive(
        "bar", [] { return BoxGeometry({0.18f, 0.04f, 1.2f}); });

    world.materials.push_back(MakeMat(0.20f, 0.22f, 0.26f, 0.0f, 0.8f));  // 0 grey wall
    world.materials.push_back(MakeMat(0.95f, 0.80f, 0.10f, 0.0f, 0.4f));  // 1 bright bar
    // 2 = slab material. transmission>0 => a CLEAR liquid (near-zero albedo, its look
    // is refraction+Fresnel+Beer); transmission==0 => a mid-grey opaque control slab.
    const bool clear = transmission > 0.0f;
    scene::RenderMaterial slab = clear ? MakeMat(0.02f, 0.03f, 0.04f, 0.0f, 0.05f)
                                       : MakeMat(0.55f, 0.57f, 0.60f, 0.0f, 0.3f);
    slab.transmission = transmission;
    slab.ior = 1.33f;
    slab.absorption[0] = absorption; slab.absorption[1] = absorption * 0.6f;
    slab.absorption[2] = absorption * 0.4f;
    world.materials.push_back(slab);

    render::RenderInstance wi; wi.mesh_id = wall; wi.render_material_id = 0u;
    wi.world_xform = {{0.0f, 1.6f, 0.6f}, math::Quat::Identity()};
    world.instances.push_back(wi);

    render::RenderInstance bi; bi.mesh_id = bar; bi.render_material_id = 1u;
    bi.world_xform = {{0.0f, 1.55f, 0.6f}, math::Quat::Identity()};
    world.instances.push_back(bi);

    if (with_slab) {
        const uint32_t slab_mesh = world.meshes.InternPrimitive(
            "slab" + std::to_string(slab_hy),
            [slab_hy] { return BoxGeometry({1.0f, slab_hy, 1.0f}); });
        render::RenderInstance si; si.mesh_id = slab_mesh; si.render_material_id = 2u;
        si.world_xform = {{0.0f, -0.2f, 0.6f}, math::Quat::Identity()};
        world.instances.push_back(si);
    }

    render::RenderLight light;
    light.color = {1.0f, 0.98f, 0.95f}; light.intensity = 4.0f;
    light.world_xform = {{-1.5f, -2.0f, 2.5f}, math::Quat::Identity()};
    world.lights.push_back(light);
    return world;
}

float MeanLuma(const rt::Framebuffer& fb) {
    double s = 0.0; const size_t n = fb.color.size() / 3u;
    for (size_t i = 0; i < n; ++i) {
        s += 0.2126 * fb.color[i * 3] + 0.7152 * fb.color[i * 3 + 1] +
             0.0722 * fb.color[i * 3 + 2];
    }
    return n ? static_cast<float>(s / static_cast<double>(n)) : 0.0f;
}

// Mean luma over the central column band where the slab covers the bright bar.
float CenterBandLuma(const rt::Framebuffer& fb) {
    double s = 0.0; size_t cnt = 0;
    const uint32_t x0 = fb.width * 3u / 8u, x1 = fb.width * 5u / 8u;
    const uint32_t y0 = fb.height / 4u, y1 = fb.height * 3u / 4u;
    for (uint32_t y = y0; y < y1; ++y) {
        for (uint32_t x = x0; x < x1; ++x) {
            const size_t i = (static_cast<size_t>(y) * fb.width + x) * 3u;
            s += 0.2126 * fb.color[i] + 0.7152 * fb.color[i + 1] + 0.0722 * fb.color[i + 2];
            ++cnt;
        }
    }
    return cnt ? static_cast<float>(s / static_cast<double>(cnt)) : 0.0f;
}

// transmission==0 NO-OP GATE: a fixed-seed beauty render of the OPAQUE test world.
// Prints an FNV checksum of the color buffer so a clean-tree run proves the bytes
// are unchanged by this change (the opaque path takes no transmissive branch).
TEST(RenderWorldRtBeauty, OpaqueBeautyNoOpChecksum) {
    auto backend = render::CreateCudaRtBackend();
    ASSERT_NE(backend, nullptr) << "no CUDA RT backend available";

    const render::RenderWorld world = BuildTestRenderWorld();
    const rt::TwoLevelScene scene = render::RenderWorldToTwoLevelScene(world);
    const rt::PinholeCamera camera = TestCamera();
    const rt::BeautyOptions opt = FixedBeauty();

    render::RtSceneHandle* h = backend->BuildScene(scene);
    ASSERT_NE(h, nullptr);
    const rt::Framebuffer a = backend->TraceBeautyToHost(h, scene, camera, opt);
    const rt::Framebuffer b = backend->TraceBeautyToHost(h, scene, camera, opt);
    backend->FreeScene(h);

    EXPECT_TRUE(FramebuffersByteEqual(a, b)) << "fixed-seed beauty not reproducible";
    std::printf("OPAQUE_BEAUTY_COLOR_FNV1A=%llu\n",
                static_cast<unsigned long long>(Fnv1a(a.color)));
    WritePpm(a, "out/rt_opaque_beauty.ppm");
    EXPECT_GT(MeanLuma(a), 0.0f);
    // Regression anchor: opaque beauty bytes must not drift. Any change that moves
    // this is a non-no-op edit to the flat shading path and must be justified.
    EXPECT_EQ(Fnv1a(a.color), 15275657819673905153ull)
        << "opaque beauty bytes drifted from the clean-tree anchor";
}

// REFRACTION: a transmissive slab in front of a bright bar must (a) shift/alter the
// covered pixels vs both the no-slab and an opaque-slab control (bent background),
// and (b) darken with more absorption (Beer-Lambert).
TEST(RenderWorldRtBeauty, RefractionBendsBackgroundAndBeerDarkens) {
    auto backend = render::CreateCudaRtBackend();
    ASSERT_NE(backend, nullptr) << "no CUDA RT backend available";
    const rt::PinholeCamera camera = RefractCamera();
    const rt::BeautyOptions opt = FixedBeauty();

    const auto trace = [&](const render::RenderWorld& w) {
        const rt::TwoLevelScene s = render::RenderWorldToTwoLevelScene(w);
        render::RtSceneHandle* h = backend->BuildScene(s);
        const rt::Framebuffer fb = backend->TraceBeautyToHost(h, s, camera, opt);
        backend->FreeScene(h);
        return fb;
    };

    const rt::Framebuffer no_slab = trace(BuildRefractWorld(false, 0.0f, 0.0f, 0.4f));
    const rt::Framebuffer opaque  = trace(BuildRefractWorld(true, 0.0f, 0.0f, 0.4f));
    const rt::Framebuffer clear   = trace(BuildRefractWorld(true, 1.0f, 0.0f, 0.4f));
    const rt::Framebuffer tinted  = trace(BuildRefractWorld(true, 1.0f, 1.5f, 0.4f));

    WritePpm(no_slab, "out/rt_refract_no_slab.ppm");
    WritePpm(opaque,  "out/rt_refract_opaque_slab.ppm");
    WritePpm(clear,   "out/rt_refract_clear_liquid.ppm");
    WritePpm(tinted,  "out/rt_refract_beer_tinted.ppm");

    const size_t n = clear.color.size() / 3u;

    // (a) TRANSMITS vs hides: where the slab covers the bright bar, the clear liquid
    // shows the bar's warm chroma (R+G) through it; the opaque grey slab blocks it.
    const float clear_band  = CenterBandLuma(clear);
    const float opaque_band = CenterBandLuma(opaque);
    std::printf("CENTER_BAND_LUMA clear=%.5f opaque=%.5f no_slab=%.5f\n",
                clear_band, opaque_band, CenterBandLuma(no_slab));
    size_t clear_vs_opaque = 0;
    for (size_t i = 0; i < n; ++i) {
        const float dr = std::fabs(clear.color[i * 3] - opaque.color[i * 3]);
        const float dg = std::fabs(clear.color[i * 3 + 1] - opaque.color[i * 3 + 1]);
        const float db = std::fabs(clear.color[i * 3 + 2] - opaque.color[i * 3 + 2]);
        if (dr + dg + db > 0.06f) ++clear_vs_opaque;
    }
    std::printf("CLEAR_VS_OPAQUE_DIFF_PIXELS=%zu of %zu\n", clear_vs_opaque, n);
    EXPECT_GT(clear_vs_opaque, n / 20u)
        << "clear liquid must look different from an opaque slab (it transmits)";

    // (a') Refraction BENDS: the clear-slab image differs from the no-slab image in
    // the covered region (the bar is shifted/distorted), not a passthrough copy.
    size_t diff = 0;
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(clear.color[i * 3 + 1] - no_slab.color[i * 3 + 1]);
        if (d > 0.02f) ++diff;
    }
    std::printf("REFRACT_DIFF_PIXELS=%zu of %zu\n", diff, n);
    EXPECT_GT(diff, n / 100u) << "clear slab must visibly refract (bend) the background";

    // (b) Beer-Lambert: more absorption darkens the transmitted band.
    const float clear_l = CenterBandLuma(clear), tinted_l = CenterBandLuma(tinted);
    std::printf("BEER center clear=%.5f tinted=%.5f\n", clear_l, tinted_l);
    EXPECT_LT(tinted_l, clear_l * 0.9f) << "Beer-Lambert must darken the thicker tint";
}

// SMOOTH NORMALS: a transmissive sphere refracts the bright-bar background. With
// per-vertex smooth normals supplied the bend differs from the faceted flat fallback
// -- this is the only test that exercises SmoothWorldNormal's barycentric blend.
TEST(RenderWorldRtBeauty, SmoothNormalsAlterRefraction) {
    auto backend = render::CreateCudaRtBackend();
    ASSERT_NE(backend, nullptr) << "no CUDA RT backend available";
    const rt::PinholeCamera camera = RefractCamera();
    const rt::BeautyOptions opt = FixedBeauty();

    const auto build = [](bool smooth) {
        render::RenderWorld w;
        const uint32_t wall = w.meshes.InternPrimitive(
            "sn_wall", [] { return BoxGeometry({2.5f, 0.05f, 1.6f}); });
        const uint32_t bar = w.meshes.InternPrimitive(
            "sn_bar", [] { return BoxGeometry({0.18f, 0.04f, 1.2f}); });
        const uint32_t ball = w.meshes.InternPrimitive(
            smooth ? "sn_ball_smooth" : "sn_ball_flat", [smooth] {
                render::MeshGeometry g = SphereGeometry(0.6f, 12u, 16u);
                if (smooth) g.normals = render::SmoothNormals(g.positions, g.indices);
                return g;
            });
        w.materials.push_back(MakeMat(0.20f, 0.22f, 0.26f, 0.0f, 0.8f));  // wall
        w.materials.push_back(MakeMat(0.95f, 0.80f, 0.10f, 0.0f, 0.4f));  // bar
        scene::RenderMaterial glass = MakeMat(0.02f, 0.03f, 0.04f, 0.0f, 0.05f);
        glass.transmission = 1.0f; glass.ior = 1.45f;
        w.materials.push_back(glass);

        render::RenderInstance wi; wi.mesh_id = wall; wi.render_material_id = 0u;
        wi.world_xform = {{0.0f, 1.6f, 0.6f}, math::Quat::Identity()};
        w.instances.push_back(wi);
        render::RenderInstance bi; bi.mesh_id = bar; bi.render_material_id = 1u;
        bi.world_xform = {{0.0f, 1.55f, 0.6f}, math::Quat::Identity()};
        w.instances.push_back(bi);
        render::RenderInstance si; si.mesh_id = ball; si.render_material_id = 2u;
        si.world_xform = {{0.0f, 0.0f, 0.6f}, math::Quat::Identity()};
        w.instances.push_back(si);

        render::RenderLight light;
        light.color = {1.0f, 0.98f, 0.95f}; light.intensity = 4.0f;
        light.world_xform = {{-1.5f, -2.0f, 2.5f}, math::Quat::Identity()};
        w.lights.push_back(light);
        return w;
    };

    const auto trace = [&](const render::RenderWorld& w) {
        const rt::TwoLevelScene s = render::RenderWorldToTwoLevelScene(w);
        render::RtSceneHandle* h = backend->BuildScene(s);
        const rt::Framebuffer fb = backend->TraceBeautyToHost(h, s, camera, opt);
        backend->FreeScene(h);
        return fb;
    };

    const rt::Framebuffer flat   = trace(build(false));
    const rt::Framebuffer smooth = trace(build(true));
    WritePpm(flat,   "out/rt_normals_flat.ppm");
    WritePpm(smooth, "out/rt_normals_smooth.ppm");

    ASSERT_EQ(flat.color.size(), smooth.color.size());
    const size_t n = smooth.color.size() / 3u;
    size_t diff = 0;
    for (size_t i = 0; i < n; ++i) {
        const float dr = std::fabs(smooth.color[i * 3] - flat.color[i * 3]);
        const float dg = std::fabs(smooth.color[i * 3 + 1] - flat.color[i * 3 + 1]);
        const float db = std::fabs(smooth.color[i * 3 + 2] - flat.color[i * 3 + 2]);
        if (dr + dg + db > 0.05f) ++diff;
    }
    std::printf("SMOOTH_VS_FLAT_DIFF_PIXELS=%zu of %zu\n", diff, n);
    EXPECT_GT(diff, n / 100u)
        << "per-vertex smooth normals must alter the refraction vs the flat fallback";
}

// OPAQUE SMOOTH NORMALS: a coarse opaque sphere shaded with smooth_normals on must
// differ from the flat-normal default -- exercises the gated ShadeBeautySmooth arm.
TEST(RenderWorldRtBeauty, OpaqueSmoothNormalsAlterShading) {
    auto backend = render::CreateCudaRtBackend();
    ASSERT_NE(backend, nullptr) << "no CUDA RT backend available";

    render::RenderWorld w;
    const uint32_t ball = w.meshes.InternPrimitive("osn_ball", [] {
        render::MeshGeometry g = SphereGeometry(0.8f, 12u, 16u);
        g.normals = render::SmoothNormals(g.positions, g.indices);
        return g;
    });
    w.materials.push_back(MakeMat(0.80f, 0.30f, 0.20f, 0.0f, 0.5f));  // matte opaque
    render::RenderInstance si; si.mesh_id = ball; si.render_material_id = 0u;
    si.world_xform = {{0.0f, 1.5f, 0.6f}, math::Quat::Identity()};
    w.instances.push_back(si);
    render::RenderLight light;
    light.color = {1.0f, 0.98f, 0.95f}; light.intensity = 4.0f;
    light.world_xform = {{-1.5f, -2.0f, 2.5f}, math::Quat::Identity()};
    w.lights.push_back(light);

    const rt::TwoLevelScene scene = render::RenderWorldToTwoLevelScene(w);
    const rt::PinholeCamera camera = RefractCamera();
    rt::BeautyOptions flat_opt = FixedBeauty();   flat_opt.smooth_normals = false;
    rt::BeautyOptions smooth_opt = FixedBeauty(); smooth_opt.smooth_normals = true;

    render::RtSceneHandle* h = backend->BuildScene(scene);
    ASSERT_NE(h, nullptr);
    const rt::Framebuffer flat   = backend->TraceBeautyToHost(h, scene, camera, flat_opt);
    const rt::Framebuffer smooth = backend->TraceBeautyToHost(h, scene, camera, smooth_opt);
    backend->FreeScene(h);
    WritePpm(flat,   "out/rt_opaque_flat.ppm");
    WritePpm(smooth, "out/rt_opaque_smooth.ppm");

    ASSERT_EQ(flat.color.size(), smooth.color.size());
    const size_t n = smooth.color.size() / 3u;
    size_t diff = 0;
    for (size_t i = 0; i < n; ++i) {
        const float dr = std::fabs(smooth.color[i * 3] - flat.color[i * 3]);
        const float dg = std::fabs(smooth.color[i * 3 + 1] - flat.color[i * 3 + 1]);
        const float db = std::fabs(smooth.color[i * 3 + 2] - flat.color[i * 3 + 2]);
        if (dr + dg + db > 0.02f) ++diff;
    }
    std::printf("OPAQUE_SMOOTH_VS_FLAT_DIFF_PIXELS=%zu of %zu\n", diff, n);
    EXPECT_GT(diff, n / 100u)
        << "smooth_normals must alter opaque shading (faceting removed) vs the flat arm";
}

// A 2-channel (grey+alpha) albedo texture must sample as grey on the device --
// channel 0 replicated to RGB -- and never read past its buffer's last texel. A
// white base material makes the albedo AOV equal the sampled texture value.
TEST(RenderWorldRt, TwoChannelTextureSamplesAsGreyNoOverread) {
    auto backend = render::CreateCudaRtBackend();
    ASSERT_NE(backend, nullptr) << "no CUDA RT backend available";

    // 4x4 constant grey+alpha: ch0 = the truth grey, ch1 = a decoy alpha the old
    // sampler mis-read as G (and over-read B one float past the last texel).
    constexpr float kGrey = 0.6f, kAlpha = 0.15f;
    rt::Texture tex;
    tex.width = 4u; tex.height = 4u; tex.channels = 2u; tex.srgb = 0u;
    tex.texels.resize(4u * 4u * 2u);
    for (uint32_t t = 0; t < 16u; ++t) {
        tex.texels[t * 2u + 0u] = kGrey;
        tex.texels[t * 2u + 1u] = kAlpha;
    }

    rt::TwoLevelScene scene;
    scene.meshes.push_back(BoxBlas({0.6f, 0.6f, 0.6f}));
    rt::Material m; m.albedo = {1.0f, 1.0f, 1.0f}; m.metallic = 0.0f; m.roughness = 0.6f;
    m.albedo_tex = 0; m.triplanar = 1u; m.uv_scale = 1.0f;
    scene.materials = {m, rt::Material{}};
    scene.instances.push_back({0u, {{0.0f, 1.5f, 0.6f}, math::Quat::Identity()}, 0u});
    scene.textures = {tex};
    scene.light.directional = false;
    scene.light.position = {-1.5f, -2.0f, 2.5f};
    scene.light.color = {1.0f, 0.98f, 0.95f};
    scene.light.intensity = 4.0f;

    const rt::PinholeCamera camera = RefractCamera();
    const rt::BeautyOptions opt = FixedBeauty();
    render::RtSceneHandle* h = backend->BuildScene(scene);
    ASSERT_NE(h, nullptr);
    const rt::Framebuffer fb = backend->TraceBeautyToHost(h, scene, camera, opt);
    backend->FreeScene(h);

    size_t hits = 0, grey_ok = 0;
    for (size_t i = 0; i < fb.prim.size(); ++i) {
        if (fb.prim[i] == rt::kNoPrim) continue;
        ++hits;
        const float r = fb.albedo[i * 3 + 0];
        const float gg = fb.albedo[i * 3 + 1];
        const float bb = fb.albedo[i * 3 + 2];
        if (std::fabs(r - kGrey) < 1e-4f && std::fabs(gg - kGrey) < 1e-4f &&
            std::fabs(bb - kGrey) < 1e-4f) ++grey_ok;
    }
    std::printf("TWO_CHANNEL_ALBEDO_HITS=%zu GREY_OK=%zu (decoy_alpha=%.3f)\n",
                hits, grey_ok, static_cast<double>(kAlpha));
    ASSERT_GT(hits, 0u) << "the textured box must cover some pixels";
    EXPECT_EQ(grey_ok, hits)
        << "2-channel albedo must sample as grey (ch0 replicated), not (grey,alpha,overread)";
}
