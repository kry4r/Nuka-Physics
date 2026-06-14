#pragma once
// ---------------------------------------------------------------------------
// nuka::render::VulkanRasterRenderer -- offscreen, real-triangle Vulkan forward
// raster renderer (M8 manifest #2; the single largest net-new lift of M8).
//
// Unlike the existing `vulkan_renderer.*` (a COMPUTE-only 2D debug-draw path:
// instance apiVersion 1.0, one compute dispatch, no graphics pipeline), this is
// a genuine GRAPHICS pipeline: VkRenderPass (1 color R8G8B8A8_UNORM + 1 depth
// D32_SFLOAT) + offscreen VkImage color/depth + VkFramebuffer + a
// VkGraphicsPipeline (mesh.vert / mesh_pbr.frag) drawing every RenderInstance's
// triangle geometry with a per-instance push-constant MVP, depth-tested.
//
// It consumes a render::RenderWorld (host triangle geometry + per-instance world
// transforms + PBR materials + cameras) and produces a VulkanOffscreenReport
// (the SAME struct the compute path emits) so the M8 render_physics_parity gate
// (T7) can `memcmp` the raw `pixels` for byte-exact determinism (G2) and check
// `non_background_pixel_count > 0` (G3).
//
// DETERMINISM (G2): the pipeline is pinned fully deterministic -- no MSAA
// (1 sample), fixed depth compare (LESS_OR_EQUAL), deterministic clear, single
// graphics queue + single submit + vkQueueWaitIdle, no time/random inputs. Two
// Render() calls on the same RenderWorld produce byte-identical pixels (verified
// on the lavapipe/llvmpipe ICD; see the smoke test).
//
// T3a SHADING is MINIMAL (flat/Lambert: one default directional light + ambient,
// base_color passed through). FULL PBR (all RenderMaterial fields +
// RenderWorld.lights + the 3 textures + tonemap) is T3b and binds in
// mesh_pbr.frag behind the seam documented there.
//
// HOST-ONLY / zero-CUDA-token (recon §4.5): no triple-chevron kernel launch, no
// cuda-runtime call, no cuda_runtime include, no phi backend_cuda include. This
// is pure C++/Vulkan.
// ---------------------------------------------------------------------------

#include "render/render_world.hpp"
#include "render/vulkan_offscreen_types.hpp"  // VulkanRgba8, VulkanOffscreenReport (shared output)

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace nuka::render {

// ---------------------------------------------------------------------------
// Vulkan handle aliases at the API boundary (M8.5 T2 present seam).
//
// The offscreen path names NO Vulkan types in its public header (it stays a pure
// data->report contract). The M8.5 present sibling (vulkan_present_renderer) +
// the ImGui InitInfo wiring need the renderer's persistent Vulkan handles, so we
// expose them through a NARROW accessor seam (RendererVulkanHandles below) WITHOUT
// dragging <vulkan/vulkan.h> into every includer. Vulkan dispatchable handles are
// opaque pointers; we forward-declare the *_T tags and alias them, ABI-identical
// to the real handles for a TU that also includes <vulkan/vulkan.h>. A non-Vulkan
// includer still sees the (offscreen) Render() API unchanged.
// ---------------------------------------------------------------------------
struct VkInstance_T;
struct VkPhysicalDevice_T;
struct VkDevice_T;
struct VkQueue_T;
struct VkRenderPass_T;
struct VkDescriptorPool_T;
struct VkSurfaceKHR_T;

using NkVkInstance       = VkInstance_T*;
using NkVkPhysicalDevice = VkPhysicalDevice_T*;
using NkVkDevice         = VkDevice_T*;
using NkVkQueue          = VkQueue_T*;
using NkVkRenderPass     = VkRenderPass_T*;
using NkVkDescriptorPool = VkDescriptorPool_T*;
// VkSurfaceKHR is a non-dispatchable handle: a 64-bit value (a pointer on
// 64-bit builds). We carry it as a void* at this seam to stay header-light; the
// present renderer reinterpret_casts it back to VkSurfaceKHR in its .cpp.
using NkVkSurface        = void*;
// VkDescriptorSetLayout / VkDescriptorSet are non-dispatchable handles too (the
// interop SSBO's set-1 layout + bound set, INT-F1). Carried as void* like
// NkVkSurface; the present renderer reinterpret_casts them back in its .cpp.
using NkVkDescriptorSetLayout = void*;
using NkVkDescriptorSet       = void*;

// ---------------------------------------------------------------------------
// OffscreenOverlayFn (M8.5 T3) -- an OPTIONAL per-render ImGui (or any extra)
// recording callback for the offscreen path's GATE-B composite. When supplied to
// Render() it is invoked INSIDE the offscreen render pass (after the scene draw,
// immediately before vkCmdEndRenderPass) with the active command buffer as an
// opaque handle (the GATE-B test reinterpret_casts it to VkCommandBuffer for
// ImGui_ImplVulkan_RenderDrawData). Mirrors PresentRenderer's OverlayRecordFn so
// the same imgui_layer recording feeds both the windowed present pass and the
// deterministic offscreen composite.
//
// ★ G2 RED-LINE: when the callback is EMPTY (the default every existing caller
// passes) Render() records a BYTE-FOR-BYTE identical command stream to the M8
// offscreen oracle -- the overlay branch is skipped entirely, so the G2/
// render_physics_parity determinism is provably unperturbed. The overlay is an
// opt-in composite for the viewer-frame smoke ONLY.
// ---------------------------------------------------------------------------
using OffscreenOverlayFn = std::function<void(void* command_buffer)>;

// ---------------------------------------------------------------------------
// RendererVulkanHandles -- the narrow accessor seam (T2/T3).
//
// Filled by VulkanRasterRenderer::VulkanHandles(). It exposes exactly the
// persistent handles the present renderer + ImGui InitInfo need (instance /
// physical device / device / queue family + queue / the offscreen render pass /
// a descriptor pool created with FREE_DESCRIPTOR_SET_BIT) -- and NOTHING of the
// private Impl. The descriptor pool is created lazily on first VulkanHandles()
// call (the offscreen path never needs one, so the offscreen-only ctor stays
// allocation-identical until a present consumer asks).
// ---------------------------------------------------------------------------
struct RendererVulkanHandles {
    NkVkInstance       instance         = nullptr;
    NkVkPhysicalDevice physical_device  = nullptr;
    NkVkDevice         device           = nullptr;
    uint32_t           graphics_family  = 0;
    uint32_t           present_family   = 0;   // == graphics_family when offscreen
    NkVkQueue          graphics_queue   = nullptr;
    NkVkQueue          present_queue    = nullptr;
    NkVkRenderPass     offscreen_render_pass = nullptr;
    NkVkDescriptorPool imgui_descriptor_pool = nullptr;  // FREE_DESCRIPTOR_SET_BIT
    uint32_t           api_version      = 0;   // the instance apiVersion (1.1)
};

// ---------------------------------------------------------------------------
// RendererConfig -- construction-time configuration (M8.5 T2).
//
// The DEFAULT (present_capable=false, surface=nullptr) reproduces the M8
// offscreen renderer BIT-FOR-BIT: instance created with ZERO extensions, device
// created with ZERO extensions, the first graphics queue selected with NO present
// check. This keeps the G2 determinism oracle pristine.
//
// When present_capable is true the renderer additionally enables the surface +
// xcb-surface INSTANCE extensions and the swapchain DEVICE extension, and -- when
// a `surface` is supplied -- selects a queue family that supports BOTH graphics
// AND present on that surface (vkGetPhysicalDeviceSurfaceSupportKHR). The present
// renderer (vulkan_present_renderer) creates the surface FIRST, then constructs
// the renderer with present_capable=true + that surface so the present-queue
// check is honoured.
// ---------------------------------------------------------------------------
struct RendererConfig {
    bool        present_capable = false;     // enable surface/swapchain ext + present queue
    NkVkSurface surface         = nullptr;   // optional probe surface for present-queue selection
};

// ---------------------------------------------------------------------------
// RasterOptions -- per-render configuration for the raster path.
//
// `camera_*_override` lets a caller (the gate / a fixed demo shot) pin a camera
// independent of RenderWorld.cameras. When `use_camera_override` is false the
// renderer uses RenderWorld.cameras[0] if present, else auto-frames the scene
// AABB with a default look-at so SOMETHING is always visible (Decision D3 keeps
// the gate from depending on authored cameras).
// ---------------------------------------------------------------------------
struct RasterOptions {
    uint32_t   width      = 1280;
    uint32_t   height     = 720;
    VulkanRgba8 background = {10, 12, 16, 255};

    // Optional explicit camera (eye/target/up + vertical FOV in degrees). When
    // use_camera_override is true these win over RenderWorld.cameras.
    bool        use_camera_override = false;
    math::Vec3  camera_eye          = {0.0f, 0.0f, 0.0f};
    math::Vec3  camera_target       = {0.0f, 0.0f, 0.0f};
    math::Vec3  camera_up           = {0.0f, 0.0f, 1.0f};
    float       camera_fov_degrees  = 45.0f;
    float       camera_near         = 0.05f;
    float       camera_far          = 1000.0f;

    // ----- BEAUTY (M8.5 T4b): grounded look + hero framing -------------------
    // A tasteful ground plane the robot sits on (a large dark disc with a soft
    // radial fade toward the horizon), drawn RENDERER-SIDE under the scene at the
    // bottom of its AABB -- NOT part of RenderWorld (keeps the RenderWorld a pure
    // data product / the M11 path-tracer + the determinism gates unaffected).
    //
    // DEFAULT OFF so the gated synthetic-scene smokes (render_raster_smoke /
    // render_physics_parity) render the EXACT same instance set as before (their
    // D1 memcmp + non_bg assertions are unperturbed). The viewer + beauty exe turn
    // it ON. With a fixed scene+camera the ground draw is itself deterministic.
    bool        draw_ground = false;

    // When draw_ground is on and the caller leaves auto-framing on (no camera
    // override and no authored camera), the renderer frames the scene as a 3/4
    // hero shot (robot filling ~60-70% of the frame, slight level/down gaze). The
    // beauty/viewer path enables this; the gates keep their plain auto-frame.
    bool        hero_framing = false;
};

// ---------------------------------------------------------------------------
// VulkanRasterRenderer -- owns the persistent Vulkan device + graphics pipeline.
//
// Construction stands up the instance/device/graphics-queue/command-pool/render
// pass/pipeline ONCE (throws std::runtime_error on any Vulkan failure -- callers
// in test/gate code catch this to GTEST_SKIP when no Vulkan device is present).
// Render() may be called repeatedly; each call rebuilds only the per-frame
// vertex/index/framebuffer resources for the requested size and returns a fresh
// report. The renderer is move-only (it owns Vulkan handles).
// ---------------------------------------------------------------------------
class VulkanRasterRenderer {
public:
    // Offscreen-only construction (M8): zero extra extensions, graphics-only
    // queue, offscreen render pass. The G2 determinism oracle -- unchanged.
    VulkanRasterRenderer();

    // Present-capable / configurable construction (M8.5 T2). With the default
    // RendererConfig{} this is byte-equivalent to the offscreen ctor above; with
    // present_capable=true it enables the surface/swapchain extensions and (given
    // a surface) a graphics+present queue. The present renderer uses this.
    explicit VulkanRasterRenderer(const RendererConfig& config);

    ~VulkanRasterRenderer();

    VulkanRasterRenderer(VulkanRasterRenderer&&) noexcept;
    VulkanRasterRenderer& operator=(VulkanRasterRenderer&&) noexcept;
    VulkanRasterRenderer(const VulkanRasterRenderer&) = delete;
    VulkanRasterRenderer& operator=(const VulkanRasterRenderer&) = delete;

    // Render `world` into an offscreen color+depth target and read the color back
    // into report.pixels (row-major, top-left origin, RGBA8). Deterministic:
    // identical (world, options) -> byte-identical report.pixels.
    //
    // `overlay` (M8.5 T3, default EMPTY) is an optional record callback run INSIDE
    // the render pass just before EndRenderPass -- used by the GATE-B viewer-frame
    // smoke to composite the ImGui UI onto the offscreen scene. When empty (every
    // M8 caller), the recorded command stream is byte-identical to the M8 oracle
    // (the G2 red-line): the overlay branch is not entered. With a FIXED ImGui UI
    // state the composited result is itself deterministic (two renders memcmp==0).
    VulkanOffscreenReport Render(const RenderWorld& world, const RasterOptions& options = {},
                                 const OffscreenOverlayFn& overlay = {});

    // The selected ICD device name (e.g. "llvmpipe (LLVM 12.0.0, 256 bits)"),
    // available after construction -- useful for the de-risk report.
    const std::string& DeviceName() const;

    // The narrow accessor seam (M8.5 T2/T3). Returns the persistent Vulkan
    // handles the present renderer + ImGui InitInfo need. The first call lazily
    // creates the FREE_DESCRIPTOR_SET_BIT descriptor pool for ImGui (so the
    // offscreen-only path never allocates it). Safe to call repeatedly (the pool
    // is created once). The handles remain valid for the renderer's lifetime.
    RendererVulkanHandles VulkanHandles();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Convenience one-shot: construct a renderer, render once, return the report.
// Throws std::runtime_error if no Vulkan graphics device is available.
VulkanOffscreenReport RenderWorldVulkanRaster(const RenderWorld& world,
                                              const RasterOptions& options = {});

}  // namespace nuka::render
