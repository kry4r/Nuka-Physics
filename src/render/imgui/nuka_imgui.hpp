#pragma once
// ---------------------------------------------------------------------------
// nuka::render::imgui -- the nuka-side Dear ImGui glue (M8.5 T1).
//
// This is the THIN wiring surface between nuka's Vulkan renderer and the
// vendored Dear ImGui (`external/imgui`, v1.92.8-docking). It owns NO Vulkan
// handles: the caller (the M8.5 T2 present renderer / T3 viewer) creates the
// instance/device/queue/render-pass/descriptor-pool and hands them in via
// NukaImGuiInitInfo. This header is deliberately Vulkan-header-light at the API
// boundary (only the handle typedefs it must name) so it can be included from
// the viewer TUs without dragging the full vendored ImGui headers into them.
//
// WHAT THIS DOES (and does NOT do):
//   - DOES wrap ImGui::CreateContext + ImGui_ImplVulkan_Init/NewFrame/
//     RenderDrawData/Shutdown into a single RAII NukaImGuiContext.
//   - DOES apply the custom-beautified "Nuka" theme (ApplyNukaTheme) + load the
//     vendored JetBrains Mono fonts -- the OWNER custom-UI requirement (NOT stock
//     ImGui). The palette/rounding/spacing is documented in nuka_imgui.cpp.
//   - DOES NOT own the platform (xcb) backend: feeding io.DisplaySize /
//     io.DeltaTime / mouse+keyboard is T2's self-written xcb backend (D1). The
//     seam is documented at NewFrame() below.
//   - DOES NOT create the swapchain/render-pass/descriptor-pool: those arrive in
//     NukaImGuiInitInfo from the present renderer (T2/T3).
//
// GATING: this lib (nuka_imgui) is compiled ONLY when NK_BUILD_VULKAN_VALIDATION
// is ON (M8.5 D3). The default build-cuda128 config never sees it.
//
// HOST-ONLY / zero-CUDA-token: pure C++/Vulkan glue; no CUDA. The vendored ImGui
// is consumed as SYSTEM includes so it is out of the zero-CUDA lint scope.
// ---------------------------------------------------------------------------

#include <cstdint>

// We name a handful of Vulkan handle types at the API boundary but do NOT want to
// force <vulkan/vulkan.h> on every includer of this header. Vulkan dispatchable
// handles are pointers; non-dispatchable ones are 64-bit. We forward-declare the
// few we expose so a caller that already includes <vulkan/vulkan.h> stays ABI
// compatible (these are the same underlying types), and a caller that does not
// can still see the struct layout. The .cpp includes the real <vulkan/vulkan.h>.
struct VkInstance_T;
struct VkPhysicalDevice_T;
struct VkDevice_T;
struct VkQueue_T;
struct VkCommandBuffer_T;
struct VkRenderPass_T;
struct VkDescriptorPool_T;

using NukaVkInstance       = VkInstance_T*;
using NukaVkPhysicalDevice = VkPhysicalDevice_T*;
using NukaVkDevice         = VkDevice_T*;
using NukaVkQueue          = VkQueue_T*;
using NukaVkCommandBuffer  = VkCommandBuffer_T*;
using NukaVkRenderPass     = VkRenderPass_T*;
using NukaVkDescriptorPool = VkDescriptorPool_T*;

namespace nuka::render::imgui {

// ---------------------------------------------------------------------------
// NukaImGuiInitInfo -- the flat handle bundle the caller fills from its Vulkan
// renderer. Mirrors the fields of `ImGui_ImplVulkan_InitInfo` that matter for an
// MSAA-1, single-subpass forward pass (the nuka present path). Internally Init()
// maps these into the upstream nested `ImGui_ImplVulkan_PipelineInfo PipelineInfoMain`
// (render-pass/subpass/MSAA moved there in imgui 1.92's 2025/09/26 reshuffle) --
// so the caller is insulated from that upstream restructure.
//
// The caller (T2/T3) supplies:
//   - instance/physical_device/device/queue_family/queue: the present renderer's
//     own handles (the M8 raster renderer holds all of these in its Impl; T2 adds
//     the accessor seam).
//   - descriptor_pool: a pool created by the caller with
//     VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT (imgui requires this; the
//     M8 raster renderer currently has NO descriptor pool, so T2/T3 create one).
//   - min_image_count / image_count: from the swapchain ( >= 2 ).
//   - render_pass + subpass: the pass imgui draws its overlay into (the swapchain
//     present pass for T3; the offscreen pass for the GATE-B overlay smoke).
//   - api_version: the instance apiVersion (the M8 renderer uses 1.1).
// ---------------------------------------------------------------------------
struct NukaImGuiInitInfo {
    uint32_t             api_version      = 0;  // e.g. VK_API_VERSION_1_1; 0 -> Init defaults it to 1.0
    NukaVkInstance       instance         = nullptr;
    NukaVkPhysicalDevice physical_device  = nullptr;
    NukaVkDevice         device           = nullptr;
    uint32_t             queue_family     = 0;
    NukaVkQueue          queue            = nullptr;
    NukaVkDescriptorPool descriptor_pool  = nullptr;  // FREE_DESCRIPTOR_SET_BIT pool from the caller
    uint32_t             min_image_count  = 2;
    uint32_t             image_count      = 2;
    NukaVkRenderPass     render_pass      = nullptr;   // the pass imgui draws into
    uint32_t             subpass          = 0;
    // MSAA is pinned to 1 sample (the nuka forward pass is single-sample for the
    // G2 determinism red-line); exposed only so a future caller could raise it.
    uint32_t             msaa_samples     = 1;  // VkSampleCountFlagBits value; 1 == VK_SAMPLE_COUNT_1_BIT
};

// ---------------------------------------------------------------------------
// NukaImGuiContext -- RAII wrapper around the ImGui context + the Vulkan backend.
//
// Lifecycle:
//   NukaImGuiContext ctx;
//   ctx.Init(info);                 // CreateContext + ApplyNukaTheme + fonts +
//                                   //   ImGui_ImplVulkan_Init
//   ... per frame:
//   ctx.NewFrame();                 // ImGui_ImplVulkan_NewFrame + ImGui::NewFrame
//   ... [T2 xcb backend has already set io.DisplaySize/DeltaTime/mouse] ...
//   ... [T3 panels: ImGui::Begin/.../End] ...
//   ImGui::Render();                // caller calls (produces ImDrawData)
//   ctx.RenderDrawData(cmd_buf);    // ImGui_ImplVulkan_RenderDrawData, INSIDE the
//                                   //   caller's BeginRenderPass..EndRenderPass
//   ... at teardown:
//   ctx.Shutdown();                 // or just let the destructor run it.
//
// Init() returns false (and leaves the context un-inited) if ImGui_ImplVulkan_Init
// fails -- the caller GTEST_SKIPs / aborts the viewer gracefully (same posture as
// the raster renderer's throw-on-no-device).
//
// Move-only (it owns the global-ish ImGui context lifetime).
// ---------------------------------------------------------------------------
class NukaImGuiContext {
public:
    NukaImGuiContext() = default;
    ~NukaImGuiContext();

    NukaImGuiContext(NukaImGuiContext&&) noexcept;
    NukaImGuiContext& operator=(NukaImGuiContext&&) noexcept;
    NukaImGuiContext(const NukaImGuiContext&) = delete;
    NukaImGuiContext& operator=(const NukaImGuiContext&) = delete;

    // Stand up the ImGui context, apply the Nuka theme + fonts, and init the
    // Vulkan backend against the caller's handles. Returns true on success.
    // Safe to call once; a second Init() without an intervening Shutdown() is a
    // no-op returning false.
    bool Init(const NukaImGuiInitInfo& info);

    // Begin a new imgui frame. NOTE: the caller (T2 xcb backend) must have already
    // populated io.DisplaySize and io.DeltaTime (and mouse/keyboard) for THIS frame
    // BEFORE NewFrame() -- this glue does NOT own the platform backend (D1). For the
    // headless GATE-B overlay smoke, the caller sets io.DisplaySize directly.
    void NewFrame();

    // Record imgui's draw data into `command_buffer`. MUST be called between the
    // caller's vkCmdBeginRenderPass and vkCmdEndRenderPass (the pass passed in
    // NukaImGuiInitInfo.render_pass). The caller must have called ImGui::Render()
    // first. No-op if not inited.
    void RenderDrawData(NukaVkCommandBuffer command_buffer);

    // Tear down the Vulkan backend + destroy the ImGui context. Idempotent.
    void Shutdown();

    // VIEW-5: rebind the ImGui Vulkan backend to a (re)created swapchain render
    // pass. After PresentFrameResult::Recreated the present renderer destroys + re-
    // creates its VkRenderPass, leaving the ImGui backend pipeline bound to a STALE
    // pass (a validation error / no-draw at best). This re-inits ONLY the Vulkan
    // backend half (ImGui_ImplVulkan_Shutdown + _Init) against `info.render_pass` /
    // min/image counts WITHOUT destroying the ImGui context -- so the docking
    // layout, theme, fonts, and all panel state survive the resize. The font atlas
    // is re-uploaded lazily on the next NewFrame (>=1.92 texture lifecycle). No-op
    // (returns false) if not yet initialized. The caller passes the SAME
    // NukaImGuiInitInfo it built from the present renderer's CURRENT handles.
    bool RebuildForRenderPass(const NukaImGuiInitInfo& info);

    bool IsInitialized() const { return initialized_; }

private:
    bool initialized_ = false;
};

// ---------------------------------------------------------------------------
// ApplyNukaTheme -- the custom-beautified "Nuka" look (the OWNER UI requirement).
//
// Applies a cohesive dark palette with a single teal accent, modern rounded
// panels, and tuned spacing to the CURRENT ImGui context's style (call after
// ImGui::CreateContext). Distinctive -- explicitly NOT the stock dark/gray theme.
// The exact palette + rounding/spacing values are documented in nuka_imgui.cpp so
// T3's panels stay consistent. NukaImGuiContext::Init calls this for you.
//
// Standalone (no Vulkan) -- usable by a pure-ImGui compile/theme check.
// ---------------------------------------------------------------------------
void ApplyNukaTheme();

// Load the vendored JetBrains Mono fonts (UI size + heading size) into the current
// context's font atlas. Returns true if at least the UI font loaded; falls back to
// the ImGui default font (AddFontDefault) and returns false if the vendored TTF is
// unreadable. NukaImGuiContext::Init calls this for you (before the backend init,
// so the atlas is ready). Standalone-safe (no Vulkan); the atlas is uploaded to the
// GPU lazily by imgui_impl_vulkan on first NewFrame (>=1.92 texture lifecycle).
bool LoadNukaFonts();

}  // namespace nuka::render::imgui
