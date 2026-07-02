#pragma once
// ---------------------------------------------------------------------------
// nuka::render::FramebufferToReport -- the HOST, CUDA-FREE glue that maps the
// ray tracer's linear-float rt::Framebuffer into the backend-agnostic
// render::VulkanOffscreenReport (RGBA8) the offscreen consumers (WritePpm /
// the parity gate) already eat. This is the path-tracer counterpart of the
// raster renderer's framebuffer download: the lavapipe raster path encodes its
// HDR result with a Narkowicz ACES filmic tonemap + sRGB OETF in mesh_pbr.frag;
// this applies the SAME tonemap + OETF on the CPU so the two backends read
// consistently. Missed primary rays (no surface hit) are filled with the caller
// background color and excluded from non_background_pixel_count.
//
// HOST-ONLY: no CUDA, no Vulkan type leaks. Compiles into nuka_render.
// ---------------------------------------------------------------------------

#include "render/vulkan_offscreen_types.hpp"
#include "rt/framebuffer.hpp"

namespace nuka::render {

// Convert a linear-float rt::Framebuffer to an RGBA8 VulkanOffscreenReport.
// `background` paints the pixels whose primary ray missed all geometry; those
// pixels are NOT counted in non_background_pixel_count. The tonemap + sRGB
// encode mirror the raster shader so the GPU-RT and lavapipe outputs read
// consistently. `keep_miss_color` true tonemaps the traced radiance for miss
// pixels too (an HDR environment backdrop) instead of painting `background`;
// false (default) is byte-identical to today.
VulkanOffscreenReport FramebufferToReport(const rt::Framebuffer& frame,
                                          VulkanRgba8 background,
                                          bool keep_miss_color = false);

}  // namespace nuka::render
