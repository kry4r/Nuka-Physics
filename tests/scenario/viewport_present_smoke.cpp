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
#include "render/raster/present_surface.hpp"
#include "render/render_world.hpp"
#include "render/window/window_surface.hpp"
#include "runtime/app/viewer/debug_draw.hpp"
#include "runtime/app/viewer/imgui_layer.hpp"
#include "runtime/app/viewer/camera_controller.hpp"
#include "render/imgui/nuka_imgui.hpp"
#include "render/viewer_resources.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_map.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include <filesystem>
#include <set>

#include <cstdint>
#include <fstream>
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

struct CaptureImage {
    uint32_t width = 0u, height = 0u;
    std::vector<unsigned char> rgb;
};

CaptureImage ReadCapture(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    CaptureImage image;
    std::string magic; int maximum = 0;
    input >> magic >> image.width >> image.height >> maximum;
    if (!input || magic != "P6" || maximum != 255) throw std::runtime_error("Invalid present capture");
    input.get();
    image.rgb.resize(static_cast<size_t>(image.width) * image.height * 3u);
    if (!input.read(reinterpret_cast<char*>(image.rgb.data()), static_cast<std::streamsize>(image.rgb.size())))
        throw std::runtime_error("Truncated present capture");
    return image;
}

class ControlledSurface final : public nuka::render::window::WindowSurface {
public:
    explicit ControlledSurface(std::unique_ptr<WindowSurface> surface)
        : surface_(std::move(surface)), width_(surface_->Width()), height_(surface_->Height()) {}

    nuka::render::window::WindowVkSurface CreateSurface(
        nuka::render::window::WindowVkInstance instance) override {
        return surface_->CreateSurface(instance);
    }
    void PollEvents(std::vector<nuka::render::window::WindowEvent>& events) override {
        surface_->PollEvents(events);
        if (resize_pending_) {
            nuka::render::window::WindowEvent event;
            event.type = nuka::render::window::WindowEvent::Type::Resize;
            event.width = width_;
            event.height = height_;
            events.push_back(event);
            resize_pending_ = false;
        }
    }
    uint32_t Width() const override { return width_; }
    uint32_t Height() const override { return height_; }
    const char* BackendName() const override { return surface_->BackendName(); }
    std::vector<std::string> RequiredInstanceExtensions() const override {
        return surface_->RequiredInstanceExtensions();
    }
    void SetExtent(uint32_t width, uint32_t height) {
        width_ = width;
        height_ = height;
        resize_pending_ = true;
    }

private:
    std::unique_ptr<WindowSurface> surface_;
    uint32_t width_;
    uint32_t height_;
    bool resize_pending_ = false;
};

}  // namespace

TEST(ViewportSurface, NegotiatesOptionalCapabilitiesAndZeroExtent) {
    using namespace nuka::render::detail;
    VkSurfaceCapabilitiesKHR caps{};
    caps.currentExtent = {~uint32_t(0), ~uint32_t(0)};
    caps.minImageExtent = {16u, 16u};
    caps.maxImageExtent = {4096u, 4096u};
    EXPECT_EQ(ChooseSwapchainExtent(caps, 0u, 720u).width, 0u);
    EXPECT_EQ(ChooseSwapchainExtent(caps, 1280u, 0u).height, 0u);
    EXPECT_EQ(ChooseSwapchainExtent(caps, 8u, 8000u).width, 16u);
    EXPECT_EQ(ChooseSwapchainExtent(caps, 8u, 8000u).height, 4096u);
    caps.currentExtent = {0u, 0u};
    EXPECT_EQ(ChooseSwapchainExtent(caps, 1280u, 720u).width, 0u);
    EXPECT_EQ(ChooseCompositeAlpha(VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR),
              VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR);
    EXPECT_EQ(ChooseCompositeAlpha(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR |
                                  VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR),
              VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
    EXPECT_THROW(ChooseCompositeAlpha(0u), std::runtime_error);
    const auto color_only = SwapchainImageUsage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    EXPECT_EQ(color_only, static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT));
    EXPECT_FALSE(CanCaptureSwapchain(color_only, VK_FORMAT_B8G8R8A8_UNORM));
    const auto capture = SwapchainImageUsage(color_only | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    EXPECT_TRUE(CanCaptureSwapchain(capture, VK_FORMAT_B8G8R8A8_UNORM));
    EXPECT_FALSE(CanCaptureSwapchain(capture, VK_FORMAT_R16G16B16A16_SFLOAT));
}

TEST(ViewportPresentSmoke, AcquireDrawPresentLoopCompletes) {
    RenderWorld world = BuildSyntheticWorld();

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

    auto controlled = std::make_unique<ControlledSurface>(std::move(surface));
    ControlledSurface* control = controlled.get();
    std::unique_ptr<nuka::render::PresentRenderer> present;
    try {
        present = std::make_unique<nuka::render::PresentRenderer>(std::move(controlled));
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

    const char* capture_path = "/tmp/nuka_present_lifecycle.ppm";
    std::remove(capture_path);
    if (present->Report().capture_supported) present->SetCaptureFrame(0, capture_path);
    constexpr int kFrames = 32;
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

    EXPECT_EQ(present->Report().frames_presented, static_cast<uint64_t>(kFrames));
    if (present->Report().capture_supported) {
        std::ifstream capture(capture_path, std::ios::binary | std::ios::ate);
        ASSERT_TRUE(capture.good());
        EXPECT_GT(capture.tellg(), static_cast<std::streamoff>(options.width) * options.height * 3);
    }
    std::vector<nuka::render::window::WindowEvent> events;
    for (int cycle = 0; cycle < 8; ++cycle) {
        control->SetExtent(0u, 0u);
        events.clear();
        present->PollEvents(events);
        const auto before = present->Report().frames_presented;
        for (int i = 0; i < 3; ++i) {
            EXPECT_EQ(present->DrawFrame(world, options), nuka::render::PresentFrameResult::Suspended);
        }
        EXPECT_EQ(present->Report().frames_presented, before);
        control->SetExtent(cycle % 2 ? 1280u : 800u, cycle % 2 ? 720u : 450u);
        events.clear();
        present->PollEvents(events);
        EXPECT_EQ(present->DrawFrame(world, options), nuka::render::PresentFrameResult::Recreated);
        EXPECT_EQ(present->DrawFrame(world, options), nuka::render::PresentFrameResult::Presented);
        EXPECT_EQ(present->Report().frames_presented, before + 1u);
    }

    nuka::runtime::app::viewer::DebugDrawBatch debug;
    nuka::scene::RenderMaterial green;
    green.base_color[0] = 0.15f; green.base_color[1] = 0.85f; green.base_color[2] = 0.25f;
    green.emissive[0] = 0.1f; green.emissive[1] = 0.4f; green.emissive[2] = 0.1f;
    green.opacity = 0.5f;
    world.materials.push_back(green);
    const float sphere[4] = {0.4f, 0.0f, 0.0f, 0.0f};
    const float box[4] = {0.25f, 0.25f, 0.3f, 0.0f};
    debug.AppendCollider(world, 0u, sphere, world.instances[0].world_xform, 2u);
    debug.AppendCollider(world, 2u, box, world.instances[1].world_xform, 2u);
    debug.AppendContact(world, {0.0f, 0.0f, 0.1f}, 2u);
    ASSERT_EQ(world.debug_instances.size(), 3u);
    const char* debug_capture = "/tmp/nuka_debug_proxies.ppm";
    std::remove(debug_capture);
    if (present->Report().capture_supported)
        present->SetCaptureFrame(static_cast<int>(present->Report().frames_presented), debug_capture);
    EXPECT_EQ(present->DrawFrame(world, options), nuka::render::PresentFrameResult::Presented);
    if (present->Report().capture_supported) {
        std::ifstream capture(debug_capture, std::ios::binary | std::ios::ate);
        ASSERT_TRUE(capture.good());
        EXPECT_GT(capture.tellg(), static_cast<std::streamoff>(options.width) * options.height * 3);
    }
    world.debug_instances.clear();
    debug.Reset();
    EXPECT_EQ(present->DrawFrame(world, options), nuka::render::PresentFrameResult::Presented);
    EXPECT_EQ(world.instances.size(), 2u);

    if (present->Report().capture_supported) {
        options.scene_viewport = {true, 100u, 80u, 640u, 400u};
        present->SetCaptureFrame(static_cast<int>(present->Report().frames_presented), "/tmp/nuka_viewport_offset.ppm");
        ASSERT_EQ(present->DrawFrame(world, options), nuka::render::PresentFrameResult::Presented);
        const auto image = ReadCapture("/tmp/nuka_viewport_offset.ppm");
        uint32_t outside = 0u, inside = 0u;
        for (uint32_t y = 0u; y < image.height; ++y) for (uint32_t x = 0u; x < image.width; ++x) {
            const size_t at = (static_cast<size_t>(y) * image.width + x) * 3u;
            const bool colored = image.rgb[at] != options.background.r || image.rgb[at + 1u] != options.background.g ||
                                 image.rgb[at + 2u] != options.background.b;
            if (!colored) continue;
            if (x >= 100u && x < 740u && y >= 80u && y < 480u) ++inside;
            else ++outside;
        }
        EXPECT_EQ(outside, 0u);
        EXPECT_GT(inside, 100u);
        options.scene_viewport.width = 0u;
        present->SetCaptureFrame(static_cast<int>(present->Report().frames_presented), "/tmp/nuka_viewport_empty.ppm");
        ASSERT_EQ(present->DrawFrame(world, options), nuka::render::PresentFrameResult::Presented);
        const auto empty = ReadCapture("/tmp/nuka_viewport_empty.ppm");
        uint32_t unexpected = 0u;
        for (size_t i = 0u; i < empty.rgb.size(); i += 3u)
            if (empty.rgb[i] != options.background.r || empty.rgb[i + 1u] != options.background.g ||
                empty.rgb[i + 2u] != options.background.b) ++unexpected;
        EXPECT_EQ(unexpected, 0u);
    }

    const auto& report = present->Report();
    std::printf("[viewport_present_smoke] backend=%s device=%s images=%u "
                "format=%d present_mode=%d %ux%u frames_presented=%llu\n",
                report.backend_name.c_str(), report.device_name.c_str(),
                report.swapchain_image_count, report.surface_format, report.present_mode,
                report.width, report.height,
                static_cast<unsigned long long>(report.frames_presented));

    present->WaitIdle();
}

TEST(ViewerWorkspace, RealLabAssetsRenderInTheModernViewport) {
    namespace viewer = nuka::runtime::app::viewer;
    namespace render = nuka::render;
    const auto path = render::ViewerResource("examples/assets/nuka_lab/gripper.nks",
                                             NUKA_SOURCE_DIR "/examples/assets/nuka_lab/gripper.nks");
    EXPECT_EQ(std::filesystem::path(path).lexically_normal(),
              (render::ViewerExecutableDirectory() / "examples/assets/nuka_lab/gripper.nks").lexically_normal());
    const auto scene = nuka::scene::nks::Load(path);
    const RenderWorld world = render::BuildRenderWorld(scene.Ecs(), nuka::scene::SceneMap{});
    ASSERT_GT(world.InstanceCount(), 10u);
    ASSERT_FALSE(world.cameras.empty());
    for (int variant = 0; variant < 4; ++variant) {
        const ImVec2 sizes[] = {ImVec2(1280.0f, 720.0f), ImVec2(960.0f, 600.0f), ImVec2(1920.0f, 1080.0f), ImVec2(1920.0f, 1080.0f)};
        const ImVec2 size = sizes[variant];
        const float dpi = variant == 3 ? 1.5f : 1.0f;
        render::window::SurfaceBackendKind kind = render::window::SurfaceBackendKind::None;
        auto surface = render::window::MakeSurface("Nuka Lab preview", static_cast<uint32_t>(size.x),
                                                    static_cast<uint32_t>(size.y), &kind);
        if (!surface) GTEST_SKIP() << "No window surface available";
        render::PresentRenderer present(std::move(surface));
        const auto vk = present.VulkanHandles();
        render::imgui::NukaImGuiInitInfo info;
        info.api_version = vk.api_version;
        info.instance = reinterpret_cast<NukaVkInstance>(vk.instance);
        info.physical_device = reinterpret_cast<NukaVkPhysicalDevice>(vk.physical_device);
        info.device = reinterpret_cast<NukaVkDevice>(vk.device);
        info.queue_family = vk.graphics_family;
        info.queue = reinterpret_cast<NukaVkQueue>(vk.graphics_queue);
        info.descriptor_pool = reinterpret_cast<NukaVkDescriptorPool>(vk.imgui_descriptor_pool);
        info.min_image_count = present.MinImageCount();
        info.image_count = present.SwapchainImageCount();
        info.render_pass = reinterpret_cast<NukaVkRenderPass>(vk.offscreen_render_pass);
        render::imgui::NukaImGuiContext imgui;
        ASSERT_TRUE(imgui.Init(info));
        viewer::ImGuiLayer ui;
        ui.EnableDocking();
        viewer::ViewerUiState state;
        state.has_scene = true;
        state.loaded_path = path;
        viewer::ViewerStats stats;
        stats.session_note = variant == 3 ? "Nuka Dynamics Lab | 实验室预览 / physics not running" : "Nuka Dynamics Lab | render snapshot, physics not running";
        viewer::CameraController camera;
        ASSERT_TRUE(camera.UseSceneCamera(world));
        render::RasterOptions options;
        options.width = present.Report().width; options.height = present.Report().height;
        const auto background = scene.Environment().sky.background;
        const auto byte = [](float v) { return static_cast<uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
        options.background = {byte(background.x), byte(background.y), byte(background.z), 255u};
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(size.x / dpi, size.y / dpi);
        io.DisplayFramebufferScale = ImVec2(dpi, dpi);
        io.DeltaTime = 1.0f / 60.0f;
        const std::string capture = "/tmp/nuka_modern_lab_" + std::to_string(options.width) + (variant == 3 ? "_dpi150.ppm" : ".ppm");
        if (present.Report().capture_supported) present.SetCaptureFrame(5, capture);
        for (int frame = 0; frame < 6; ++frame) {
            imgui.NewFrame();
            ui.RecordUi(world, stats, camera, state, &scene, options.width, options.height);
            EXPECT_GT(ImGui::CalcTextSize("WWW").x, ImGui::CalcTextSize("iii").x * 2.0f);
            for (const ImWchar codepoint : {static_cast<ImWchar>(0x5b9e), static_cast<ImWchar>(0x9f98), static_cast<ImWchar>(0x2000b)})
                EXPECT_NE(ImGui::GetFontBaked()->FindGlyphNoFallback(codepoint), nullptr);
            options.scene_viewport = ui.Viewport().pixels;
            if (frame < 3) ASSERT_TRUE(camera.UseSceneCamera(world, ui.Viewport().Aspect(), static_cast<float>(options.width) / static_cast<float>(options.height)));
            camera.WriteOptions(options);
            ImGui::Render();
            EXPECT_EQ(present.DrawFrame(world, options, [&imgui](void* command) {
                imgui.RenderDrawData(reinterpret_cast<NukaVkCommandBuffer>(command));
            }), render::PresentFrameResult::Presented);
        }
        if (present.Report().capture_supported) {
            const auto image = ReadCapture(capture);
            const auto rect = ui.Viewport().pixels;
            std::set<uint32_t> scene_colors;
            for (uint32_t y = rect.y + 4u; y + 4u < rect.y + rect.height; y += 4u)
                for (uint32_t x = rect.x + 4u; x + 4u < rect.x + rect.width; x += 4u) {
                    const size_t at = (static_cast<size_t>(y) * image.width + x) * 3u;
                    scene_colors.insert((static_cast<uint32_t>(image.rgb[at]) << 16u) |
                                        (static_cast<uint32_t>(image.rgb[at + 1u]) << 8u) | image.rgb[at + 2u]);
                }
            EXPECT_GT(scene_colors.size(), 64u) << "Scene pixels must not be hidden by an opaque UI background";
        }
        EXPECT_TRUE(ui.Viewport().Valid());
        EXPECT_GT(ui.Viewport().pixels.x, 0u);
        EXPECT_LT(ui.Viewport().pixels.width, options.width);
        EXPECT_TRUE(render::imgui::HasNukaCjkFont());
        std::printf("[modern_lab] backend=%s framebuffer=%ux%u viewport=%u,%u,%u,%u font_atlas=%dx%d\n",
                    present.Report().backend_name.c_str(), options.width, options.height,
                    ui.Viewport().pixels.x, ui.Viewport().pixels.y, ui.Viewport().pixels.width,
                    ui.Viewport().pixels.height, io.Fonts->TexData->Width, io.Fonts->TexData->Height);
        present.WaitIdle();
        imgui.Shutdown();
    }
}
