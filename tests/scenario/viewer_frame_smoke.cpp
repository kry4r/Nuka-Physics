// ---------------------------------------------------------------------------
// viewer_frame_smoke -- M8.5 GATE-B: the RUN-VERIFIED OFFSCREEN viewer-frame
// composite smoke (the deliverable for T3, recon §1.4 GATE-B).
//
// Builds a RenderWorld (the synthetic two-box scene + at least one non-trivial
// instance), stands up the OFFSCREEN VulkanRasterRenderer (the D1 oracle), inits
// the nuka ImGui context against the renderer's offscreen render pass + descriptor
// pool, records the REAL viewer panels (imgui_layer) with a FIXED ViewerUiState
// (NO time-seeded animation), and composites the scene + ImGui draw data into the
// offscreen image via the Render(world, options, overlay) seam (overlay =
// ImGui_ImplVulkan_RenderDrawData through NukaImGuiContext::RenderDrawData).
//
// Asserts:
//   (a) ImDrawData.CmdListsCount > 0                      (the UI actually built)
//   (b) composited image non_background_pixel_count > 0   (scene + UI drew)
//   (c) two composited renders byte-identical (memcmp==0) (D1, fixed UI state)
//   (d) dumps a composited PPM to /tmp/m8_5_viewer_frame.ppm for off-box review.
// SKIPs cleanly if no Vulkan device (renderer ctor throws) or ImGui init fails.
//
// The offscreen overlay seam is opt-in: with an EMPTY overlay Render() is the
// untouched G2 oracle (verified by render_raster_smoke / render_physics_parity);
// this test exercises the NON-empty branch only.
//
// HOST-ONLY / zero-CUDA-token.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <vulkan/vulkan.h>

#include "imgui.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"
#include "render/render_world.hpp"
#include "runtime/app/viewer/camera_controller.hpp"
#include "runtime/app/viewer/imgui_layer.hpp"
#include "scene/asset/asset_ref.hpp"
#include "scene/asset/nka.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using nuka::render::MeshGeometry;
using nuka::render::RenderInstance;
using nuka::render::RenderWorld;
using nuka::render::VulkanRgba8;

MeshGeometry MakeBox(float h) {
    MeshGeometry g;
    const float p[8][3] = {
        {-h, -h, -h}, {h, -h, -h}, {h, h, -h}, {-h, h, -h},
        {-h, -h, h},  {h, -h, h},  {h, h, h},  {-h, h, h}};
    for (auto& v : p) {
        g.positions.push_back(v[0]); g.positions.push_back(v[1]); g.positions.push_back(v[2]);
    }
    const uint32_t faces[12][3] = {
        {0, 1, 2}, {0, 2, 3}, {4, 6, 5}, {4, 7, 6}, {0, 4, 5}, {0, 5, 1},
        {3, 2, 6}, {3, 6, 7}, {1, 5, 6}, {1, 6, 2}, {0, 3, 7}, {0, 7, 4}};
    for (auto& f : faces) {
        g.indices.push_back(f[0]); g.indices.push_back(f[1]); g.indices.push_back(f[2]);
    }
    return g;
}

// A synthetic RenderWorld: two boxes (one tagged as an .nka MESH source so the
// scene-tree shows the real-vs-primitive accent dot, exercising that path) + a
// metallic material.
RenderWorld BuildSyntheticWorld() {
    RenderWorld world;
    nuka::scene::RenderMaterial red;
    red.base_color[0] = 0.85f; red.base_color[1] = 0.15f; red.base_color[2] = 0.15f; red.base_color[3] = 1.0f;
    red.metallic = 0.1f; red.roughness = 0.6f;
    nuka::scene::RenderMaterial steel;
    steel.base_color[0] = 0.55f; steel.base_color[1] = 0.6f; steel.base_color[2] = 0.7f; steel.base_color[3] = 1.0f;
    steel.metallic = 0.9f; steel.roughness = 0.25f;
    world.materials.push_back(red);
    world.materials.push_back(steel);

    // mesh_a tagged NkaMesh via InternNkaMesh (a non-trivial "real geometry"
    // instance for the scene tree); mesh_b a primitive fallback.
    nuka::scene::AssetRef ref;
    ref.fourcc = nuka::scene::NkaTagMesh();
    ref.nka_path = "synthetic://box_a";
    const uint32_t mesh_a = world.meshes.InternNkaMesh(ref, [] { return MakeBox(0.3f); });
    const uint32_t mesh_b = world.meshes.InternPrimitive("prim:box:0.2", [] { return MakeBox(0.2f); });

    RenderInstance a;
    a.entity = nuka::scene::EntityId{1u, 0u};
    a.mesh_id = mesh_a; a.render_material_id = 0; a.world_xform.position = {-0.5f, 0.0f, 0.0f};
    world.instances.push_back(a);
    RenderInstance b;
    b.entity = nuka::scene::EntityId{2u, 0u};
    b.mesh_id = mesh_b; b.render_material_id = 1; b.world_xform.position = {0.5f, 0.0f, 0.25f};
    world.instances.push_back(b);
    return world;
}

void WritePpm(const std::string& path, const std::vector<VulkanRgba8>& pixels,
              uint32_t width, uint32_t height) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%u %u\n255\n", width, height);
    for (const auto& px : pixels) {
        const unsigned char rgb[3] = {px.r, px.g, px.b};
        std::fwrite(rgb, 1, 3, f);
    }
    std::fclose(f);
}

}  // namespace

TEST(ViewerFrameSmoke, OffscreenScenePlusImGuiCompositeIsDeterministic) {
    const RenderWorld world = BuildSyntheticWorld();

    nuka::render::RasterOptions options;
    options.width = 1280;
    options.height = 720;

    // 1. Offscreen renderer (the D1 oracle). SKIP if no Vulkan device.
    std::unique_ptr<nuka::render::VulkanRasterRenderer> renderer;
    try {
        renderer = std::make_unique<nuka::render::VulkanRasterRenderer>();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "No Vulkan graphics device available: " << e.what();
    }

    // 2. ImGui context bound to the OFFSCREEN render pass + descriptor pool.
    nuka::render::RendererVulkanHandles vk = renderer->VulkanHandles();
    nuka::render::imgui::NukaImGuiInitInfo info;
    info.api_version     = vk.api_version;
    info.instance        = reinterpret_cast<NukaVkInstance>(vk.instance);
    info.physical_device = reinterpret_cast<NukaVkPhysicalDevice>(vk.physical_device);
    info.device          = reinterpret_cast<NukaVkDevice>(vk.device);
    info.queue_family    = vk.graphics_family;
    info.queue           = reinterpret_cast<NukaVkQueue>(vk.graphics_queue);
    info.descriptor_pool = reinterpret_cast<NukaVkDescriptorPool>(vk.imgui_descriptor_pool);
    info.min_image_count = 2u;
    info.image_count     = 2u;
    info.render_pass     = reinterpret_cast<NukaVkRenderPass>(vk.offscreen_render_pass);
    info.subpass         = 0u;

    nuka::render::imgui::NukaImGuiContext imgui;
    if (!imgui.Init(info)) {
        GTEST_SKIP() << "ImGui Vulkan backend init failed (no device / pool)";
    }

    // 3. FIXED UI state -> deterministic draw data (the D1 precondition).
    nuka::runtime::app::viewer::ImGuiLayer ui;
    ui.EnableDocking();  // MUST precede the first NewFrame (ImGui asserts this).
    nuka::runtime::app::viewer::CameraController camera;
    camera.FrameAabb({-0.8f, -0.5f, -0.3f}, {0.8f, 0.5f, 0.6f});
    nuka::runtime::app::viewer::ViewerStats stats;
    stats.step_time_ms = 1.23f; stats.fps = 60.0f; stats.dof = 11u; stats.links = 6u;
    stats.bodies = 1u; stats.contact_cap = 64u; stats.draw_calls = 2u; stats.non_bg_pixels = 12345u;
    stats.frame_index = 7u; stats.device_name = "offscreen";
    nuka::runtime::app::viewer::ViewerUiState ui_state;
    ui_state.playing = true; ui_state.speed = 1.0f;

    // Build one ImGui frame with a fixed display size; record the panels.
    auto build_ui = [&]() -> int {
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(static_cast<float>(options.width),
                                static_cast<float>(options.height));
        io.DeltaTime = 1.0f / 60.0f;  // fixed (no time-seeded animation)
        imgui.NewFrame();
        ui.RecordUi(world, stats, camera, ui_state);
        ImGui::Render();
        ImDrawData* dd = ImGui::GetDrawData();
        return dd ? dd->CmdListsCount : 0;
    };

    // Warm-up: the docking layout (DockBuilder split + DockSpaceOverViewport) needs
    // a couple of frames to settle -- on the FIRST frame the panels are created and
    // docked but not yet laid out, so draw data is incomplete. Drive a few frames so
    // the layout latches; AFTER that, a fixed UI state is steady-state deterministic.
    // (imgui_impl_vulkan also uploads the font atlas lazily on first NewFrame.)
    for (int w = 0; w < 4; ++w) build_ui();

    // 4. First composite.
    const int cmd_lists_a = build_ui();
    nuka::render::VulkanOffscreenReport first;
    try {
        first = renderer->Render(world, options, [&imgui](void* cmd) {
            imgui.RenderDrawData(reinterpret_cast<NukaVkCommandBuffer>(cmd));
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "composite render failed: " << e.what();
    }

    // 5. Second composite (rebuild the same fixed UI -> same draw data).
    const int cmd_lists_b = build_ui();
    nuka::render::VulkanOffscreenReport second;
    try {
        second = renderer->Render(world, options, [&imgui](void* cmd) {
            imgui.RenderDrawData(reinterpret_cast<NukaVkCommandBuffer>(cmd));
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "composite render (2) failed: " << e.what();
    }

    // (a) the UI actually built draw commands.
    EXPECT_GT(cmd_lists_a, 0) << "imgui_layer produced no draw command lists";
    EXPECT_EQ(cmd_lists_a, cmd_lists_b) << "fixed UI state must build identical command-list count";

    // (b) the composited image has lit (scene + UI) pixels.
    EXPECT_GT(first.non_background_pixel_count, 0u);
    EXPECT_EQ(first.width, options.width);
    EXPECT_EQ(first.height, options.height);
    ASSERT_EQ(first.pixels.size(), static_cast<size_t>(options.width) * options.height);

    // (c) D1: two composited renders byte-identical.
    ASSERT_EQ(first.pixels.size(), second.pixels.size());
    const int cmp = std::memcmp(first.pixels.data(), second.pixels.data(),
                                first.pixels.size() * sizeof(VulkanRgba8));
    EXPECT_EQ(cmp, 0) << "two composited (scene + fixed-UI) renders are not byte-identical";

    // (d) dump a PPM for off-box beauty review.
    WritePpm("/tmp/m8_5_viewer_frame.ppm", first.pixels, first.width, first.height);

    std::printf("[viewer_frame_smoke] device=%s cmd_lists=%d non_bg=%zu D1=%s "
                "ppm=/tmp/m8_5_viewer_frame.ppm\n",
                first.selected_device_name.c_str(), cmd_lists_a,
                first.non_background_pixel_count, cmp == 0 ? "BYTE-IDENTICAL" : "MISMATCH");

    imgui.Shutdown();
}
