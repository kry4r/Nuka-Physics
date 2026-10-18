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
#include <vector>

namespace nuka::render {

// The result of one DrawFrame call.
enum class PresentFrameResult : uint8_t {
    Presented,   // acquire -> draw -> present succeeded this frame
    Recreated,   // swapchain rebuilt; a suboptimal frame may have been presented
    Error,       // an unrecoverable Vulkan error -- caller should stop the loop
    Suspended,   // framebuffer has zero extent; keep pumping events until restored
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
    bool        capture_supported = false;
};

// An optional per-frame ImGui (or any extra) recording callback. It is invoked
// INSIDE the swapchain render pass (after the scene draw, before EndRenderPass)
// with the active command buffer as an opaque handle (the viewer reinterpret_casts
// it to VkCommandBuffer for ImGui_ImplVulkan_RenderDrawData). May be empty.
using OverlayRecordFn = std::function<void(void* command_buffer)>;

// One frame's pose slot from the CUDA pose path: a set-1 descriptor set of per-instance
// mat4 rows, readable once `timeline` reaches `ready_value`.
struct PresentInterop {
    NkVkDescriptorSet set = nullptr;
    NkVkSemaphore timeline = nullptr;
    uint64_t ready_value = 0u;
};

// The world instances an id pass saw in a framebuffer region of the scene viewport; rows past
// `instances` are `env_instances` copies.
struct PickResult {
    uint32_t center = ~0u;               // the instance under the region's centre, or ~0u
    std::vector<uint32_t> instances;     // every instance in the region, ascending
};

// The most scene views one DrawViews frame draws.
inline constexpr uint32_t kMaxSceneViews = 4u;

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
    // OUT_OF_DATE/SUBOPTIMAL by recreating the swapchain (returns Recreated). With
    // `interop`, instance transforms come from its pose slot instead of `world`.
    PresentFrameResult DrawFrame(const RenderWorld& world,
                                 const RasterOptions& options = {},
                                 const OverlayRecordFn& overlay = {},
                                 const PresentInterop* interop = nullptr);

    // DrawFrame over up to kMaxSceneViews views, each into its own scene_viewport; the
    // first view's background, instance flags and outline colours apply to the frame.
    PresentFrameResult DrawViews(const RenderWorld& world, const std::vector<RasterOptions>& views,
                                 const OverlayRecordFn& overlay, const PresentInterop* interop = nullptr);

    // Builds the instanced pipeline against the pose slots' set layout (set 1); null
    // drops it. The caller keeps the layout alive while it is installed.
    void SetInteropTransforms(NkVkDescriptorSetLayout ssbo_set_layout);

    // Renders `world` into the id pass as DrawFrame would and reads back `region` (framebuffer
    // pixels); kInstanceUnpickable instances are seen through. Waits for the GPU.
    PickResult Pick(const RenderWorld& world, const RasterOptions& options, const SceneViewport& region,
                    const PresentInterop* interop = nullptr);

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
    // A submitted frame's resources are free again once FramesInFlight() later frames
    // were submitted; SubmittedFrames() counts submissions.
    uint32_t FramesInFlight() const;
    uint64_t SubmittedFrames() const;

    const PresentReport& Report() const;
    const std::string&   DeviceName() const;

    // Pump the window's event queue (close/resize/input) into `out` (appended).
    void PollEvents(std::vector<window::WindowEvent>& out);

    // True until a Close event has been observed (the viewer's loop condition).
    bool ShouldClose() const;

    // Block until the device is idle (call before destroying external resources
    // that the in-flight frames may reference, e.g. the ImGui backend).
    void WaitIdle();

    // Capture the TRUE presented pixels of DrawFrame index `frame_index` (0-based) to
    // a binary PPM at `ppm_path` (swapchain readback, bypasses DWM/WGC). <0 disables.
    void SetCaptureFrame(int frame_index, std::string ppm_path);
    // Capture only the scene viewport of DrawFrame `frame_index`, as drawn before the overlay.
    void SetSceneCaptureFrame(int frame_index, std::string ppm_path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nuka::render
