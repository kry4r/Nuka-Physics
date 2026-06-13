#pragma once
// ---------------------------------------------------------------------------
// nuka::render::vulkan_renderer -- DEBUG OVERLAY renderer (M8 T8 demotion).
//
// ROLE (post-M8): this is the SECONDARY / DEBUG-OVERLAY render path. It is a
// COMPUTE-only 2D wireframe / debug-draw pass (orthographic outlines of bodies,
// contacts, frames, AABBs) intended to be layered as an OVERLAY on top of the
// real shaded image -- NOT to produce the primary picture.
//
// The PRIMARY offscreen render path is now the forward-PBR triangle rasterizer
// `render/raster/vulkan_raster_renderer.*` (M8 T3a; real VkRenderPass +
// VkGraphicsPipeline + depth + PBR materials). Whole-scene "render this scene"
// requests should go through THAT renderer; this file only draws debug overlays.
//
// To make the demotion explicit the public entry points are now named with an
// `Overlay` / `Debug` intent:
//   * ProbeVulkanOverlayBackend  -- probe the compute device for the overlay path
//   * RenderDebugOverlayVulkan   -- rasterize a debug-draw command list (overlay)
//   * RenderSceneDebugOverlayVulkan -- draw a RenderScene as debug wireframe overlay
// The pre-demotion names (ProbeVulkanRenderer / RenderDebugDrawListVulkan /
// RenderSceneVulkan) are retained as thin compatibility aliases so existing
// callers + the compute-path tests keep working unchanged.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "render/render_scene.hpp"
#include "render/vulkan_offscreen_types.hpp"  // RenderBackend, VulkanRgba8, VulkanOffscreenReport

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::render {

// Backend probe report for the DEBUG-OVERLAY (compute) path.
struct VulkanOverlayBackendReport {
    RenderBackend backend = RenderBackend::Vulkan;
    bool production_backend = true;
    uint32_t instance_api_version_major = 0;
    uint32_t instance_api_version_minor = 0;
    uint32_t instance_api_version_patch = 0;
    uint32_t physical_device_count = 0;
    std::string selected_device_name;
};

// Compatibility alias for the pre-demotion name.
using VulkanRendererReport = VulkanOverlayBackendReport;

enum class VulkanDebugDrawCommandType : uint32_t {
    Line = 0,
    Sphere = 1,
    Capsule = 2,
    Box = 3,
    AABB = 4,
    Frame = 5,
    ContactPoint = 6
};

struct VulkanDebugDrawCommand {
    VulkanDebugDrawCommandType type = VulkanDebugDrawCommandType::Line;
    math::Vec3 position = math::Vec3::Zero();
    math::Vec3 end = math::Vec3::Zero();
    math::Vec3 size = math::Vec3::Zero();
    float radius = 0.0f;
    float half_height = 0.0f;
    uint32_t color = 0xFFFFFFFFu;
};

struct VulkanOffscreenOptions {
    uint32_t width = 640;
    uint32_t height = 360;
    float view_scale = 180.0f;
    math::Vec3 view_center = {0.0f, 0.0f, 0.25f};
    VulkanRgba8 background = {10, 12, 16, 255};
};

// -- DEBUG-OVERLAY entry points (the demoted, secondary render path) ---------

// Probe the compute device backing the debug-overlay path.
VulkanOverlayBackendReport ProbeVulkanOverlayBackend();

// Rasterize a debug-draw command list into an offscreen overlay image.
VulkanOffscreenReport RenderDebugOverlayVulkan(
    const std::vector<VulkanDebugDrawCommand>& commands,
    const VulkanOffscreenOptions& options = {});

// Draw a RenderScene as a debug wireframe overlay (NOT the primary shaded image;
// for that use render/raster/vulkan_raster_renderer.*).
VulkanOffscreenReport RenderSceneDebugOverlayVulkan(
    const RenderScene& scene,
    const VulkanOffscreenOptions& options = {});

// -- Compatibility aliases for the pre-demotion names ------------------------
// Existing callers + the compute-path tests use these; they forward to the
// Overlay entry points above.
inline VulkanOverlayBackendReport ProbeVulkanRenderer() {
    return ProbeVulkanOverlayBackend();
}
inline VulkanOffscreenReport RenderDebugDrawListVulkan(
    const std::vector<VulkanDebugDrawCommand>& commands,
    const VulkanOffscreenOptions& options = {}) {
    return RenderDebugOverlayVulkan(commands, options);
}
inline VulkanOffscreenReport RenderSceneVulkan(
    const RenderScene& scene,
    const VulkanOffscreenOptions& options = {}) {
    return RenderSceneDebugOverlayVulkan(scene, options);
}

} // namespace nuka::render
