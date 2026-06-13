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
#include <memory>

namespace nuka::render {

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
    VulkanRasterRenderer();
    ~VulkanRasterRenderer();

    VulkanRasterRenderer(VulkanRasterRenderer&&) noexcept;
    VulkanRasterRenderer& operator=(VulkanRasterRenderer&&) noexcept;
    VulkanRasterRenderer(const VulkanRasterRenderer&) = delete;
    VulkanRasterRenderer& operator=(const VulkanRasterRenderer&) = delete;

    // Render `world` into an offscreen color+depth target and read the color back
    // into report.pixels (row-major, top-left origin, RGBA8). Deterministic:
    // identical (world, options) -> byte-identical report.pixels.
    VulkanOffscreenReport Render(const RenderWorld& world, const RasterOptions& options = {});

    // The selected ICD device name (e.g. "llvmpipe (LLVM 12.0.0, 256 bits)"),
    // available after construction -- useful for the de-risk report.
    const std::string& DeviceName() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Convenience one-shot: construct a renderer, render once, return the report.
// Throws std::runtime_error if no Vulkan graphics device is available.
VulkanOffscreenReport RenderWorldVulkanRaster(const RenderWorld& world,
                                              const RasterOptions& options = {});

}  // namespace nuka::render
