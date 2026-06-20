// ---------------------------------------------------------------------------
// Offline RHI replay gate: the OfflineRenderer (the host render layer over the
// PHI-unified tracer) must add ZERO numeric change vs driving render::RtBackendI
// directly. Renders ONE RenderWorld BOTH ways with a Beauty profile and asserts
// the framebuffers are BYTE-IDENTICAL across all channels (D1) -- proving the
// RHI layer is pure orchestration over the same kernel.
//
// The scene mirrors tests/rt/test_render_world_rt.cpp (floor box + cube, two
// materials, a point light) so the gate exercises the real adapter path. Built
// -ffp-contract=off to stay consistent with the rt kernel's --fmad=false.
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "render/render_world.hpp"
#include "render/rt_adapter.hpp"
#include "render/rt_backend.hpp"
#include "rhi/offline/offline_renderer.hpp"
#include "rhi/offline/render_profile.hpp"
#include "rt/camera.hpp"
#include "rt/framebuffer.hpp"
#include "rt/two_level_render.hpp"
#include "scene/ecs/components.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace nuka;

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr uint32_t kW = 128u;
constexpr uint32_t kH = 96u;

// A unit box (half-extents) as positions + indices (12 triangles), matching the
// winding tests/rt/test_render_world_rt.cpp uses so the scenes are comparable.
render::MeshGeometry BoxGeometry(math::Vec3 he) {
    const float x = he.x, y = he.y, z = he.z;
    render::MeshGeometry g;
    g.positions = {
        -x, -y, -z,  x, -y, -z,  x,  y, -z, -x,  y, -z,
        -x, -y,  z,  x, -y,  z,  x,  y,  z, -x,  y,  z,
    };
    g.indices = {
        4,5,6, 4,6,7,   0,3,2, 0,2,1,   1,2,6, 1,6,5,
        0,4,7, 0,7,3,   2,3,7, 2,7,6,   0,1,5, 0,5,4,
    };
    return g;
}

scene::RenderMaterial MakeMat(float r, float g, float b, float metal, float rough) {
    scene::RenderMaterial m;
    m.base_color[0] = r; m.base_color[1] = g; m.base_color[2] = b; m.base_color[3] = 1.0f;
    m.metallic = metal;
    m.roughness = rough;
    return m;
}

// Floor box + small box, two materials, a point light -- a non-trivial scene.
render::RenderWorld BuildTestRenderWorld() {
    render::RenderWorld world;

    const uint32_t floor_mesh =
        world.meshes.InternPrimitive("floor", [] { return BoxGeometry({2.5f, 2.5f, 0.05f}); });
    const uint32_t cube_mesh =
        world.meshes.InternPrimitive("cube", [] { return BoxGeometry({0.4f, 0.4f, 0.4f}); });

    world.materials.push_back(MakeMat(0.10f, 0.11f, 0.13f, 0.0f, 0.70f));  // floor
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

rt::PinholeCamera TestCamera() {
    return rt::BuildPinhole({-1.6f, -2.0f, 1.4f}, {0.0f, 0.0f, 0.4f},
                            {0.0f, 0.0f, 1.0f}, 0.35f * kPi, kW, kH);
}

// memcmp all 6 AOVs of two framebuffers; true iff byte-identical.
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
    for (float d : fb.depth) {
        if (d > 0.0f && d < 1.0e30f) ++hits;
    }
    return hits;
}

}  // namespace

// The OfflineRenderer beauty output equals the direct RtBackendI beauty output,
// byte-identical across all AOVs: the RHI layer adds zero numeric change (D1).
TEST(OfflineReplay, OfflineRendererBeautyMatchesDirectTracerByteExact) {
    auto direct = render::CreateCudaRtBackend();
    ASSERT_NE(direct, nullptr) << "no CUDA RT backend available";

    const render::RenderWorld world = BuildTestRenderWorld();
    const rt::PinholeCamera camera = TestCamera();

    // ONE shared set of cinematic controls used by BOTH paths (fixed seed => the
    // beauty trace is reproducible; identical inputs + identical kernel => bytes).
    rhi::offline::RenderProfile profile =
        rhi::offline::RenderProfile::BeautyReplay(kW, kH);
    const rt::BeautyOptions beauty = profile.beauty;

    // Path A: drive RtBackendI directly (the existing adapter path).
    const rt::TwoLevelScene scene = render::RenderWorldToTwoLevelScene(world);
    render::RtSceneHandle* h = direct->BuildScene(scene);
    ASSERT_NE(h, nullptr);
    const rt::Framebuffer direct_frame = direct->TraceBeautyToHost(h, scene, camera, beauty);
    direct->FreeScene(h);

    // Path B: the same world + camera + profile through the OfflineRenderer.
    rhi::offline::OfflineRenderer renderer;
    ASSERT_TRUE(renderer.usable()) << "OfflineRenderer obtained no RT backend";
    const rt::Framebuffer offline_frame = renderer.Render(world, camera, profile);

    EXPECT_EQ(offline_frame.width, kW);
    EXPECT_EQ(offline_frame.height, kH);
    EXPECT_GT(HitPixels(offline_frame), 0u) << "rendered frame has no geometry";
    EXPECT_TRUE(FramebuffersByteEqual(direct_frame, offline_frame));
}
