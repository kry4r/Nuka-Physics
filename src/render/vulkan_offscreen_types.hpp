#pragma once
// ---------------------------------------------------------------------------
// nuka::render -- shared offscreen render output types.
//
// These backend-agnostic pixel/report types are the COMMON surface that both
// offscreen renderers emit:
//   * the legacy COMPUTE debug-draw path (render/vulkan_renderer.*), and
//   * the M8 forward RASTER path (render/raster/vulkan_raster_renderer.*).
//
// They were factored out of vulkan_renderer.hpp so the raster renderer + the
// M8 render_physics_parity gate can consume `VulkanOffscreenReport::pixels` for
// the G2 byte-identical memcmp and the G3 non_background_pixel_count check
// without depending on the compute-overlay renderer. vulkan_renderer.hpp
// includes this header so the report/pixel names/definitions remain identical
// for the compute overlay path and its tests (no behavioral change).
// (The legacy render::RenderScene scene-overlay path was deleted in M11.)
//
// HOST-ONLY: no CUDA, no Vulkan type leaks here -- pure POD.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nuka::render {

enum class RenderBackend {
    Vulkan
};

struct VulkanRgba8 {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t a = 255;
};

struct VulkanOffscreenReport {
    RenderBackend backend = RenderBackend::Vulkan;
    bool production_backend = true;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t command_count = 0;
    uint32_t physical_device_count = 0;
    std::string selected_device_name;
    size_t non_background_pixel_count = 0;
    std::vector<VulkanRgba8> pixels;
};

}  // namespace nuka::render
