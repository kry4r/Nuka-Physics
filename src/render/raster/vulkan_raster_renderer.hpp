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
#include <limits>
#include <memory>
#include <string>
#include <vector>

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

    // ----- OFFLINE BEAUTY LOOK LEVERS (studio path tracer only) --------------
    // Neutral defaults keep every existing beauty frame byte-identical; the demo
    // render entry raises them per the scene's authored environment levers.
    float       beauty_sky_fill     = 0.30f;               // env-miss indirect fill scale
    float       beauty_sun_disc[3]  = {0.0f, 0.0f, 0.0f};  // sky sun-disc radiance (0 = off)
    float       beauty_exposure_ev  = 0.0f;                // post exposure in stops
    float       beauty_grade        = 0.0f;                // post contrast/saturation strength

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

    // The ground disc albedo (linear rgb). DEFAULT = the original near-black cool
    // studio floor so every existing draw_ground caller is unchanged; the M10 demo
    // tools raise it to a mid-grey studio sweep so a DARK robot foot reads against
    // it (a near-black foot on a near-black floor was the "floating" cue).
    float       ground_color[3] = {0.018f, 0.020f, 0.026f};

    // M10 grounded-look CONTACT SHADOW (0 => OFF, the default, so the gated
    // synthetic smokes are byte-identical). When > 0 AND draw_ground is on AND no
    // explicit contact_points are supplied, the renderer paints ONE soft radial
    // occlusion into the GROUND disc under the scene footprint (centre = AABB xy
    // centre, radius from the xy extent). This is the simple single-blob fallback.
    // Robot/scene draws never receive it, so their output is unperturbed (G2-safe).
    float       contact_shadow_strength = 0.0f;

    // EXPLICIT per-contact shadows (the good look): a caller (e.g. the quadruped
    // walk tool) supplies one entry PER FOOT each frame -- world xy of the contact
    // patch, its radius, and a strength it fades with foot lift. The renderer draws
    // each as a soft "shadow decal" disc on the floor (floor-coloured at the rim ->
    // dark at the centre, so it blends seamlessly with NO alpha blending). When
    // non-empty these REPLACE the single-blob fallback. Each entry = {x, y, radius,
    // strength}; an entry with radius<=0 or strength<=0 is skipped.
    // Each entry = {x, y, radius, strength}; an optional `z` places the decal at a
    // GIVEN world height (for non-flat terrain: drop the decal onto the LOCAL surface
    // under the foot rather than the single flat floor plane). DEFAULT z = a NaN
    // sentinel meaning "use the flat floor_z" -> every existing caller (e.g. the flat
    // go2_walk_video) is byte-identical: the renderer only honours a FINITE z.
    struct ContactPoint {
        float x = 0.0f, y = 0.0f, radius = 0.0f, strength = 0.0f;
        float z = std::numeric_limits<float>::quiet_NaN();  // NaN => use flat floor_z
    };
    std::vector<ContactPoint> contact_points;

    // ----- ATMOSPHERIC HAZE / FOG (Go2-terrain demo: the Isaac-Lab look) -------
    // A simple exponential distance fog that fades far geometry toward a bright
    // horizon colour. DEFAULT density 0 => OFF, so every existing caller is
    // byte-identical (the gated synthetic smokes + G2 parity are unperturbed):
    // the fog is packed into a previously-unused SceneUbo slot (ambient_ground.w),
    // and the shader's mix is a strict no-op when density == 0. When > 0 the
    // fragment colour is lerped toward `fog_color` by 1 - exp(-density * view_dist)
    // (distance from the camera eye), giving distant dogs/terrain the soft white
    // atmospheric fade. Tune density ~0.02-0.06 per metre for a gentle haze.
    float       fog_density   = 0.0f;                  // 0 => fog OFF (G2-safe default)
    float       fog_color[3]  = {0.93f, 0.95f, 0.98f}; // near-white horizon (linear rgb)

    // ----- CINEMATIC SUN LIGHTING (Go2-terrain demo: the Isaac-Lab look) -------
    // A single strong directional KEY light (the "sun") replacing the soft 3-point
    // studio rig, plus a LOW ambient fill, so vertical step risers darken vs the
    // horizontal treads (geometry reads) and a mid-grey concrete albedo stays mid-
    // grey (not washed to white). DEFAULT use_sun_light=false => the renderer keeps
    // the EXACT pre-existing 3-point rig + ambient, so the gated synthetic smokes
    // (render_raster_smoke / render_physics_parity) are byte-identical (G2-safe).
    // The demo turns it ON. When on AND world.lights is empty the renderer uses
    // ONLY this sun (+ sun_ambient_*); authored world.lights still win when present.
    //
    //   sun_direction  : world-space direction TOWARD the sun (need NOT be unit;
    //                    the renderer normalizes). A LOW elevation (small +z) casts
    //                    the long crisp shadows that carve the stairs. Default points
    //                    down from the upper-front at ~28 deg elevation.
    //   sun_color      : the sun radiance (color * intensity, linear rgb). A strong
    //                    near-white key; raise to brighten the lit treads.
    //   sun_ambient_sky/ground : the hemispheric ambient (sky overhead, ground
    //                    bounce). Kept LOW so the unlit riser faces stay dark and the
    //                    N.L contrast carves the geometry. rgb linear.
    bool        use_sun_light       = false;            // false => legacy 3-point rig (G2-safe)
    float       sun_direction[3]    = {0.42f, -0.34f, 0.50f};  // toward the sun (low-ish)
    float       sun_color[3]        = {3.05f, 2.95f, 2.78f};   // strong warm-white key
    float       sun_ambient_sky[3]  = {0.30f, 0.34f, 0.42f};   // cool sky fill (low)
    float       sun_ambient_ground[3] = {0.16f, 0.16f, 0.17f}; // ground bounce (low)

    // ----- VERTICAL SKY GRADIENT BACKGROUND -----------------------------------
    // Replace the flat `background` clear with a soft vertical gradient drawn into
    // the clear (a full-screen sky: a brighter horizon low in the frame easing to a
    // cooler darker zenith up top), the Isaac-Lab cool-grey sky. DEFAULT
    // sky_gradient=false => the flat `background` clear is used unchanged, so the
    // gated smokes clear to the SAME constant colour (G2-safe). When true the
    // renderer paints a screen-space gradient sky BEFORE the scene (a 2-triangle
    // full-screen draw with a dedicated sky pipeline; depth write off so the scene
    // overwrites it). sky_top = zenith (up), sky_bottom = horizon (down); linear rgb.
    bool        sky_gradient   = false;                       // false => flat clear (G2-safe)
    float       sky_top[3]     = {0.62f, 0.67f, 0.74f};       // cooler zenith
    float       sky_bottom[3]  = {0.84f, 0.87f, 0.91f};       // brighter horizon

    // ----- DIRECTIONAL SHADOW MAP (the hero Isaac-Lab feature) ----------------
    // A real depth-from-the-sun shadow map: the scene depth is rendered once from
    // the sun's orthographic view, then sampled (with a small PCF kernel) in
    // mesh_pbr.frag so the stairs self-shadow and the robots cast crisp ground
    // shadows into the grooves. DEFAULT shadow_strength=0 => NO shadow pass is
    // recorded, the sun-shadow descriptor/sampler is never bound, and the fragment
    // shader's shadow term is a strict no-op -> the gated smokes record a BYTE-FOR-
    // BYTE identical command stream + pixels (G2-safe). Requires use_sun_light=true
    // (the shadow is cast from the sun direction). The demo turns it ON.
    //
    //   shadow_strength : 0 => OFF (default). 1 => fully dark occlusion; ~0.7-0.9
    //                     gives crisp-but-not-pure-black Isaac-Lab grooves.
    //   shadow_map_size : the square shadow-map resolution (px). 2048 is crisp on
    //                     lavapipe offline; lower it if a render is too slow.
    //   shadow_bias     : depth bias (shadow-map units) to kill self-shadow acne on
    //                     the lit treads. Tune up if treads show moire; down if a
    //                     thin light gap ("peter-panning") appears at contact.
    float       shadow_strength  = 0.0f;     // 0 => shadow pass OFF (G2-safe default)
    uint32_t    shadow_map_size  = 2048u;    // shadow map resolution (px, square)
    float       shadow_bias      = 0.0025f;  // depth bias to suppress acne
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
