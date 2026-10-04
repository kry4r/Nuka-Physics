#pragma once

#include "render/raster/scene_viewport.hpp"

#include <algorithm>
#include <cmath>

namespace nuka::runtime::app::viewer {

// Logical coordinates are aligned to the same framebuffer pixels used by the renderer.
struct SceneViewportRect {
    float x = 0.0f, y = 0.0f, width = 0.0f, height = 0.0f;
    float scale_x = 1.0f, scale_y = 1.0f;
    render::SceneViewport pixels{true, 0u, 0u, 0u, 0u};

    bool Valid() const { return !pixels.Empty(); }
    bool Contains(float px, float py) const {
        return Valid() && px >= x && py >= y && px < x + width && py < y + height;
    }
    float LocalX(float px) const { return (px - x) * scale_x; }
    float LocalY(float py) const { return (py - y) * scale_y; }
    float Aspect() const { return pixels.Aspect(); }

    static SceneViewportRect FromLogical(float x, float y, float width, float height,
                                         float origin_x, float origin_y, float scale_x, float scale_y,
                                         uint32_t framebuffer_width, uint32_t framebuffer_height) {
        SceneViewportRect out;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(width) || !std::isfinite(height) ||
            !std::isfinite(origin_x) || !std::isfinite(origin_y) || !std::isfinite(scale_x) ||
            !std::isfinite(scale_y) || width <= 0.0f || height <= 0.0f || scale_x <= 0.0f || scale_y <= 0.0f)
            return out;
        const auto edge = [](double value, uint32_t limit, bool upper) {
            return static_cast<uint32_t>(std::clamp(upper ? std::ceil(value) : std::floor(value),
                                                     0.0, static_cast<double>(limit)));
        };
        const uint32_t left = edge((static_cast<double>(x) - origin_x) * scale_x, framebuffer_width, false);
        const uint32_t top = edge((static_cast<double>(y) - origin_y) * scale_y, framebuffer_height, false);
        const uint32_t right = edge((static_cast<double>(x) + width - origin_x) * scale_x, framebuffer_width, true);
        const uint32_t bottom = edge((static_cast<double>(y) + height - origin_y) * scale_y, framebuffer_height, true);
        out.pixels = {true, left, top, right - left, bottom - top};
        out.scale_x = scale_x; out.scale_y = scale_y;
        out.x = origin_x + static_cast<float>(left) / scale_x;
        out.y = origin_y + static_cast<float>(top) / scale_y;
        out.width = static_cast<float>(right - left) / scale_x;
        out.height = static_cast<float>(bottom - top) / scale_y;
        if (!std::isfinite(out.x) || !std::isfinite(out.y) || !std::isfinite(out.width) || !std::isfinite(out.height))
            return {};
        return out;
    }
};

}  // namespace nuka::runtime::app::viewer
