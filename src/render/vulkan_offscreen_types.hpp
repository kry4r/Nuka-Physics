#pragma once
// ---------------------------------------------------------------------------
// nuka::render -- shared offscreen render output types.
//
// These backend-agnostic pixel/report types are the COMMON surface that both
// offscreen renderers emit:
//   * the legacy COMPUTE debug-draw path (render/vulkan_renderer.*), and
//   * the M8 forward RASTER path (render/raster/vulkan_raster_renderer.*).
//
// They were factored out of vulkan_renderer.hpp (which also pulls in the legacy
// render_scene.hpp) so the raster renderer + the M8 render_physics_parity gate
// can consume `VulkanOffscreenReport::pixels` for the G2 byte-identical memcmp
// and the G3 non_background_pixel_count check WITHOUT transitively including
// render_scene.hpp -- whose RenderCamera/RenderLight names collide with the new
// render_world.hpp definitions (render_scene.* is deleted in M9). vulkan_renderer.hpp
// includes this header so the names/definitions remain identical for the
// existing compute path and its tests (no behavioral change).
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
