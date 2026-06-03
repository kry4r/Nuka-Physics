#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- multi-AOV framebuffer for the sensor renderer (v0.7 p13, entry-plan
// §4.C). The p12 renderer emitted {depth, prim_id}; p13 widens it to the full AOV
// set the sensor phase (p14b) surfaces as DLPack-able channels:
//
//   color       RGB shaded radiance (Lambert+GGX direct + ambient)   3*float / px
//   depth       world-space distance from camera (RtMissDepth on miss) float / px
//   normal      world-space surface normal (geometric)                3*float / px
//   albedo      material base color at the hit (for denoising/sim2real)3*float / px
//   uv          surface parameterisation (bary u,v / sphere uv / 0)   2*float / px
//   prim_id     ORIGINAL primitive index (semantic / instance seg)    uint32 / px
//
// LAYOUT (the p14b DLPack contract): ONE CONTIGUOUS host buffer PER CHANNEL,
// row-major (index = y*width + x), interleaved-by-component within a channel
// (color is r,g,b,r,g,b,...). Each channel is a separate std::vector so a sensor
// can DLPack-export exactly the channels it needs with no repacking. Named,
// contiguous, byte-stable (D1: produced by an atomic-free one-writer-per-pixel
// kernel; two runs memcmp-identical across ALL channels).
//
// HOST struct only (download target). The device kernel writes the same set of
// per-channel device buffers, downloaded here.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cstdint>
#include <vector>

namespace nuka::rt {

struct Framebuffer {
    uint32_t width = 0u;
    uint32_t height = 0u;

    std::vector<float>    color;   // 3 * width * height  (r,g,b interleaved)
    std::vector<float>    depth;   // 1 * width * height
    std::vector<float>    normal;  // 3 * width * height  (x,y,z interleaved)
    std::vector<float>    albedo;  // 3 * width * height  (r,g,b interleaved)
    std::vector<float>    uv;      // 2 * width * height  (u,v interleaved)
    std::vector<uint32_t> prim;    // 1 * width * height

    size_t Pixels() const {
        return static_cast<size_t>(width) * static_cast<size_t>(height);
    }

    math::Vec3 ColorAt(uint32_t x, uint32_t y) const {
        const size_t p = (static_cast<size_t>(y) * width + x) * 3u;
        return {color[p], color[p + 1u], color[p + 2u]};
    }
    math::Vec3 NormalAt(uint32_t x, uint32_t y) const {
        const size_t p = (static_cast<size_t>(y) * width + x) * 3u;
        return {normal[p], normal[p + 1u], normal[p + 2u]};
    }
    math::Vec3 AlbedoAt(uint32_t x, uint32_t y) const {
        const size_t p = (static_cast<size_t>(y) * width + x) * 3u;
        return {albedo[p], albedo[p + 1u], albedo[p + 2u]};
    }
    float DepthAt(uint32_t x, uint32_t y) const {
        return depth[static_cast<size_t>(y) * width + x];
    }
    uint32_t PrimAt(uint32_t x, uint32_t y) const {
        return prim[static_cast<size_t>(y) * width + x];
    }
    float UAt(uint32_t x, uint32_t y) const {
        return uv[(static_cast<size_t>(y) * width + x) * 2u];
    }
    float VAt(uint32_t x, uint32_t y) const {
        return uv[(static_cast<size_t>(y) * width + x) * 2u + 1u];
    }
};

} // namespace nuka::rt
