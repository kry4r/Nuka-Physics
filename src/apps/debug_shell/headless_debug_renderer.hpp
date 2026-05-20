#pragma once
// ---------------------------------------------------------------------------
// nuka::app::headless_debug_renderer -- deterministic debug draw rasterizer
// ---------------------------------------------------------------------------

#include "apps/debug_shell/debug_draw.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::app {

struct Rgba8 {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t a = 255;
};

struct DebugRasterImage {
    uint32_t width = 0;
    uint32_t height = 0;
    Rgba8 background = {10, 12, 16, 255};
    std::vector<Rgba8> pixels;

    size_t NonBackgroundPixelCount() const;
    bool WritePpm(const std::string& path) const;
};

struct DebugRasterOptions {
    uint32_t width = 640;
    uint32_t height = 360;
    float view_scale = 180.0f;
    math::Vec3 view_center = {0.0f, 0.0f, 0.25f};
    Rgba8 background = {10, 12, 16, 255};
};

DebugRasterImage RasterizeDebugDrawList(const DebugDrawList& list,
                                        const DebugRasterOptions& options = {});

} // namespace nuka::app
