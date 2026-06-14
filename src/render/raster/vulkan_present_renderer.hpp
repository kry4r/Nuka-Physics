#pragma once
// ---------------------------------------------------------------------------
// nuka::render::PresentRenderer -- the realtime SWAPCHAIN present sibling of the
// offscreen VulkanRasterRenderer (M8.5 T2, the chief net-new Vulkan lift).
//
// STRATEGY-SPLIT (recon 7, decision: do NOT mutate the offscreen path):
//   * The offscreen VulkanRasterRenderer::Render() stays the byte-deterministic
//     G2 oracle (TRANSFER_SRC finalLayout + readback + vkQueueWaitIdle), pristine.
//   * PresentRenderer is a SEPARATE class. It BORROWS a present-capable
//     VulkanRasterRenderer (constructed with RendererConfig{present_capable=true,
//     surface}) for the shared device / queue / shaders, but owns its OWN
//     swapchain render pass (finalLayout PRESENT_SRC_KHR), its own graphics
//     pipeline built against that pass, per-image framebuffers (swapchain view +
//     a shared depth), and per-frame sync (imageAvailable / renderFinished
//     semaphores + an in-flight fence, frames-in-flight = 2).
//
// The realtime present loop is LEGITIMATELY non-D1 (vsync / acquire ordering /
// frames-in-flight) -- it is isolated here and never touches the offscreen
// determinism path (recon 1.4 red-line).
//
// DEFENSIVE swapchain negotiation (recon 7, lavapipe + 1.2.131 loader): we QUERY
// surface formats / present modes / capabilities and pick (FIFO present mode; an
// SRGB or UNORM B8G8R8A8/R8G8B8A8 format as available; clamped image count). We
// hardcode NOTHING -- lavapipe caps differ from real hardware.
//
// USAGE (the viewer T3 + the GATE-C present smoke):
//   auto surface = window::MakeSurface("nuka", w, h, &kind);
//   PresentRenderer present(std::move(surface));   // throws on no device/surface
//   for (...) {
//       PresentFrameResult r = present.DrawFrame(world, options /*, imgui_record*/);
//       if (r == PresentFrameResult::Error) break;
//   }
//
// HOST-ONLY / zero-CUDA-token: pure C++ / Vulkan. No CUDA tokens.
// ---------------------------------------------------------------------------

#include "render/raster/vulkan_raster_renderer.hpp"  // RasterOptions, RendererVulkanHandles
#include "render/render_world.hpp"
#include "render/window/window_surface.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace nuka::render {

// The result of one DrawFrame call.
enum class PresentFrameResult : uint8_t {
    Presented,   // acquire -> draw -> present succeeded this frame
    Recreated,   // swapchain was OUT_OF_DATE/SUBOPTIMAL and rebuilt; no present this call
    Error,       // an unrecoverable Vulkan error -- caller should stop the loop
};

// ---------------------------------------------------------------------------
// PresentReport -- a small post-construction / post-frame fact bundle for the
// gate + de-risk report (image count, negotiated format/present-mode, backend).
// ---------------------------------------------------------------------------
struct PresentReport {
    std::string device_name;
    std::string backend_name;        // "xcb" / "headless"
    uint32_t    swapchain_image_count = 0;
    int32_t     surface_format = 0;  // VkFormat as int (for the report)
    int32_t     present_mode = 0;    // VkPresentModeKHR as int
    uint32_t    width = 0;
    uint32_t    height = 0;
    uint64_t    frames_presented = 0;
};

// An optional per-frame ImGui (or any extra) recording callback. It is invoked
// INSIDE the swapchain render pass (after the scene draw, before EndRenderPass)
// with the active command buffer as an opaque handle (the viewer reinterpret_casts
// it to VkCommandBuffer for ImGui_ImplVulkan_RenderDrawData). May be empty.
using OverlayRecordFn = std::function<void(void* command_buffer)>;

class PresentRenderer {
public:
    // Construct from a window surface (ownership transferred). Stands up the
    // present-capable device, the swapchain, render pass, pipeline, sync. Throws
    // std::runtime_error if no Vulkan device / no surface / swapchain creation
    // fails (callers GTEST_SKIP / abort the viewer gracefully).
    explicit PresentRenderer(std::unique_ptr<window::WindowSurface> surface);
    ~PresentRenderer();

    PresentRenderer(PresentRenderer&&) noexcept;
    PresentRenderer& operator=(PresentRenderer&&) noexcept;
    PresentRenderer(const PresentRenderer&) = delete;
    PresentRenderer& operator=(const PresentRenderer&) = delete;

    // Acquire -> record (draw `world` via the shared pipeline + optional overlay)
    // -> submit (per-frame semaphores + in-flight fence) -> present. Handles
    // OUT_OF_DATE/SUBOPTIMAL by recreating the swapchain (returns Recreated).
    PresentFrameResult DrawFrame(const RenderWorld& world,
                                 const RasterOptions& options = {},
                                 const OverlayRecordFn& overlay = {});

    // M11 INT-F1 (CUDA<->Vulkan zero-copy consumer): enable the OPT-IN instanced
    // draw path. When the interop publisher has stood up an exportable transform
    // SSBO (InteropTransformSsbo), the viewer passes its descriptor-set layout
    // (set 1, STORAGE_BUFFER) + the bound descriptor set here. DrawFrame then binds
    // the 2nd graphics pipeline (mesh_instanced.vert) + that SSBO at set 1 and draws
    // each instance by gl_InstanceIndex reading the SCATTERED device transforms,
    // INSTEAD of the per-draw push-constant world_xform. The DEFAULT per-draw
    // pipeline (the D1 oracle path) stays in place and is used whenever interop is
    // NOT enabled, so the default offscreen/present render is byte-unchanged.
    // Pass VK_NULL_HANDLE handles to disable (revert to the per-draw path).
    // Opaque NkVk* handles keep this header CUDA-free and Vulkan-light at the seam.
    void SetInteropTransforms(NkVkDescriptorSetLayout ssbo_set_layout,
                              NkVkDescriptorSet ssbo_set);

    // True iff SetInteropTransforms installed a usable instanced pipeline (the
    // viewer logs an HONEST "interop ACTIVE" only when this AND the publisher's
    // device scatter are both live).
    bool InteropDrawActive() const;

    // The narrow Vulkan handle seam (for the viewer's ImGui InitInfo): device /
    // queue / render pass / descriptor pool / image counts. Mirrors what
    // NukaImGuiInitInfo needs (the swapchain present pass is render_pass).
    RendererVulkanHandles VulkanHandles();

    // The swapchain present render pass (the pass ImGui must draw into for the
    // viewer overlay). Returned via the handle seam too, but exposed directly for
    // clarity. Opaque VkRenderPass.
    NkVkRenderPass PresentRenderPass() const;

    uint32_t SwapchainImageCount() const;
    uint32_t MinImageCount() const;

    const PresentReport& Report() const;
    const std::string&   DeviceName() const;

    // Pump the window's event queue (close/resize/input) into `out` (appended).
    void PollEvents(std::vector<window::WindowEvent>& out);

    // True until a Close event has been observed (the viewer's loop condition).
    bool ShouldClose() const;

    // Block until the device is idle (call before destroying external resources
    // that the in-flight frames may reference, e.g. the ImGui backend).
    void WaitIdle();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nuka::render
