// ---------------------------------------------------------------------------
// nuka::render::FramebufferToReport -- impl. HOST, CUDA-FREE. See the header for
// the contract. The ACES filmic + sRGB OETF match mesh_pbr.frag so the GPU-RT
// and lavapipe-raster outputs read consistently.
// ---------------------------------------------------------------------------

#include "render/rt_framebuffer_to_report.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nuka::render {

namespace {

// The RT miss sentinel mirrors the kernel's kNoPrim (ray_box.cuh) -- a missed
// primary ray writes this prim id; we cannot include the .cuh from this host TU.
constexpr uint32_t kRtNoPrim = 0xFFFFFFFFu;

// Narkowicz ACES filmic tonemap (matches mesh_pbr.frag AcesFilmic).
float AcesFilmic(float x) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    const float y = (x * (a * x + b)) / (x * (c * x + d) + e);
    return std::min(1.0f, std::max(0.0f, y));
}

// Standard sRGB OETF (matches mesh_pbr.frag LinearToSrgb).
float LinearToSrgb(float c) {
    if (c <= 0.0031308f) return c * 12.92f;
    return 1.055f * std::pow(std::max(c, 0.0f), 1.0f / 2.4f) - 0.055f;
}

uint8_t Encode(float linear) {
    const float v = LinearToSrgb(AcesFilmic(linear));
    return static_cast<uint8_t>(std::lround(std::min(1.0f, std::max(0.0f, v)) * 255.0f));
}

}  // namespace

VulkanOffscreenReport FramebufferToReport(const rt::Framebuffer& frame,
                                          VulkanRgba8 background,
                                          bool keep_miss_color) {
    VulkanOffscreenReport rep;
    rep.width = frame.width;
    rep.height = frame.height;
    const size_t pixels = frame.Pixels();
    rep.pixels.resize(pixels);

    const bool have_prim = frame.prim.size() == pixels;
    const bool have_color = frame.color.size() == pixels * 3u;
    size_t non_bg = 0;
    for (size_t p = 0; p < pixels; ++p) {
        const bool hit = have_prim && frame.prim[p] != kRtNoPrim;
        if ((hit || keep_miss_color) && have_color) {
            VulkanRgba8 px;
            px.r = Encode(frame.color[p * 3u + 0u]);
            px.g = Encode(frame.color[p * 3u + 1u]);
            px.b = Encode(frame.color[p * 3u + 2u]);
            px.a = 255u;
            rep.pixels[p] = px;
            if (hit) ++non_bg;
        } else {
            rep.pixels[p] = background;
        }
    }
    rep.non_background_pixel_count = non_bg;
    return rep;
}

}  // namespace nuka::render
