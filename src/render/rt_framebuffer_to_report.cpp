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

struct Rgb { float r, g, b; };

// Exposure (already a linear scale) + a gentle filmic contrast/saturation grade
// about mid-grey. grade <= 0 leaves the exposed colour untouched (identity).
Rgb GradeColor(Rgb c, float exposure_scale, float grade) {
    c.r *= exposure_scale; c.g *= exposure_scale; c.b *= exposure_scale;
    if (grade <= 0.0f) return c;
    const float pivot = 0.18f;
    const float contrast = 1.0f + 0.35f * grade;
    const float sat = 1.0f + 0.30f * grade;
    const float luma = 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
    auto ch = [&](float x) {
        x = luma + (x - luma) * sat;        // saturation about luma
        x = pivot + (x - pivot) * contrast; // contrast about mid-grey
        return x < 0.0f ? 0.0f : x;
    };
    return Rgb{ch(c.r), ch(c.g), ch(c.b)};
}

}  // namespace

VulkanOffscreenReport FramebufferToReport(const rt::Framebuffer& frame,
                                          VulkanRgba8 background,
                                          bool keep_miss_color,
                                          float exposure_ev,
                                          float grade) {
    VulkanOffscreenReport rep;
    rep.width = frame.width;
    rep.height = frame.height;
    const size_t pixels = frame.Pixels();
    rep.pixels.resize(pixels);

    // exposure_ev == 0 && grade == 0 is a strict no-op (an exact *1.0 scale then
    // an identity grade), so the encode is byte-identical to the ungraded path.
    const float exposure_scale =
        (exposure_ev == 0.0f) ? 1.0f : std::exp2(exposure_ev);
    const bool have_prim = frame.prim.size() == pixels;
    const bool have_color = frame.color.size() == pixels * 3u;
    size_t non_bg = 0;
    for (size_t p = 0; p < pixels; ++p) {
        const bool hit = have_prim && frame.prim[p] != kRtNoPrim;
        if ((hit || keep_miss_color) && have_color) {
            const Rgb c = GradeColor(
                Rgb{frame.color[p * 3u + 0u], frame.color[p * 3u + 1u],
                    frame.color[p * 3u + 2u]}, exposure_scale, grade);
            VulkanRgba8 px;
            px.r = Encode(c.r);
            px.g = Encode(c.g);
            px.b = Encode(c.b);
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
