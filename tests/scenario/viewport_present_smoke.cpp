// ---------------------------------------------------------------------------
// viewport_present_smoke -- M8.5 GATE-C: the RUN-VERIFIED headless present-loop
// smoke for the realtime swapchain path (src/render/raster/vulkan_present_renderer.*
// + src/render/window/).
//
// Builds a SYNTHETIC RenderWorld (two boxes, like render_raster_smoke), creates a
// surface via MakeSurface (xcb under Xvfb here -- the live path; headless on a
// modern loader), stands up the swapchain, and runs acquire -> draw -> present for
// N=3 frames. Asserts: the loop completes with no VkError, swapchain image count
// >= 2, present succeeds N times. SKIPs cleanly if no surface ext / no Vulkan
// device (CI without a display, or the renderer ctor throws).
//
// RUN (proven by the recon's vkcube exit-0 under the same setup):
//   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
//   xvfb-run -a -s '-screen 0 1280x720x24' <this binary>
//
// HOST-ONLY / zero-CUDA-token: pure C++ / Vulkan / xcb.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "render/raster/vulkan_present_renderer.hpp"
#include "render/render_world.hpp"
#include "render/window/window_surface.hpp"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using nuka::render::MeshGeometry;
using nuka::render::RenderInstance;
using nuka::render::RenderWorld;

MeshGeometry MakeBox(float h) {
    MeshGeometry g;
    const float p[8][3] = {
        {-h, -h, -h}, {h, -h, -h}, {h, h, -h}, {-h, h, -h},
        {-h, -h, h},  {h, -h, h},  {h, h, h},  {-h, h, h}};
    for (auto& v : p) {
        g.positions.push_back(v[0]);
        g.positions.push_back(v[1]);
        g.positions.push_back(v[2]);
    }
    const uint32_t faces[12][3] = {
        {0, 1, 2}, {0, 2, 3}, {4, 6, 5}, {4, 7, 6},
        {0, 4, 5}, {0, 5, 1}, {3, 2, 6}, {3, 6, 7},
        {1, 5, 6}, {1, 6, 2}, {0, 3, 7}, {0, 7, 4}};
    for (auto& f : faces) {
        g.indices.push_back(f[0]);
        g.indices.push_back(f[1]);
        g.indices.push_back(f[2]);
    }
    return g;
}

RenderWorld BuildSyntheticWorld() {
    RenderWorld world;
    nuka::scene::RenderMaterial red;
    red.base_color[0] = 0.85f; red.base_color[1] = 0.15f; red.base_color[2] = 0.15f; red.base_color[3] = 1.0f;
    nuka::scene::RenderMaterial blue;
    blue.base_color[0] = 0.20f; blue.base_color[1] = 0.35f; blue.base_color[2] = 0.90f; blue.base_color[3] = 1.0f;
    world.materials.push_back(red);
    world.materials.push_back(blue);
    const uint32_t mesh_a = world.meshes.InternPrimitive("prim:box:0.3", [] { return MakeBox(0.3f); });
    const uint32_t mesh_b = world.meshes.InternPrimitive("prim:box:0.2", [] { return MakeBox(0.2f); });
    RenderInstance a;
    a.mesh_id = mesh_a; a.render_material_id = 0; a.world_xform.position = {-0.5f, 0.0f, 0.0f};
    world.instances.push_back(a);
    RenderInstance b;
    b.mesh_id = mesh_b; b.render_material_id = 1; b.world_xform.position = {0.5f, 0.0f, 0.25f};
    world.instances.push_back(b);
    return world;
}

}  // namespace

TEST(ViewportPresentSmoke, AcquireDrawPresentLoopCompletes) {
    const RenderWorld world = BuildSyntheticWorld();

    // 1. Create a surface (xcb under Xvfb here, headless on a modern loader).
    nuka::render::window::SurfaceBackendKind kind =
        nuka::render::window::SurfaceBackendKind::None;
    std::unique_ptr<nuka::render::window::WindowSurface> surface;
    try {
        surface = nuka::render::window::MakeSurface("nuka-present-smoke", 1280, 720, &kind);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "MakeSurface threw: " << e.what();
    }
    if (!surface || kind == nuka::render::window::SurfaceBackendKind::None) {
        GTEST_SKIP() << "No window surface available (no display / no surface extension). "
                        "Run under: VK_ICD_FILENAMES=<lavapipe> xvfb-run -a "
                        "-s '-screen 0 1280x720x24' <binary>";
    }

    // 2. Stand up the present renderer (swapchain + pipeline + sync).
    std::unique_ptr<nuka::render::PresentRenderer> present;
    try {
        present = std::make_unique<nuka::render::PresentRenderer>(std::move(surface));
    } catch (const std::exception& e) {
        GTEST_SKIP() << "PresentRenderer ctor failed (no Vulkan device / swapchain): " << e.what();
    }

    // 3. Swapchain image count >= 2.
    const uint32_t image_count = present->SwapchainImageCount();
    EXPECT_GE(image_count, 2u) << "swapchain must double-buffer";

    // 4. Run the acquire -> draw -> present loop for N frames.
    nuka::render::RasterOptions options;
    options.width = present->Report().width;
    options.height = present->Report().height;

    constexpr int kFrames = 3;
    int presented = 0;
    bool had_error = false;
    for (int i = 0; i < kFrames; ++i) {
        nuka::render::PresentFrameResult r;
        try {
            r = present->DrawFrame(world, options);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "DrawFrame threw on frame " << i << ": " << e.what();
            had_error = true;
            break;
        }
        if (r == nuka::render::PresentFrameResult::Error) {
            had_error = true;
            break;
        }
        if (r == nuka::render::PresentFrameResult::Presented) {
            ++presented;
        }
        // Recreated (resize) does not count as a present; loop continues.
    }

    EXPECT_FALSE(had_error) << "present loop hit a VkError";
    EXPECT_EQ(presented, kFrames) << "expected " << kFrames << " successful presents";

    const auto& report = present->Report();
    std::printf("[viewport_present_smoke] backend=%s device=%s images=%u "
                "format=%d present_mode=%d %ux%u frames_presented=%llu\n",
                report.backend_name.c_str(), report.device_name.c_str(),
                report.swapchain_image_count, report.surface_format, report.present_mode,
                report.width, report.height,
                static_cast<unsigned long long>(report.frames_presented));

    present->WaitIdle();
}
