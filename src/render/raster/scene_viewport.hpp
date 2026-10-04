#pragma once

#include <algorithm>
#include <cstdint>

namespace nuka::render {

struct SceneViewport {
    bool enabled = false;
    uint32_t x = 0u, y = 0u, width = 0u, height = 0u;
    bool Empty() const { return width == 0u || height == 0u; }
    float Aspect() const { return Empty() ? 1.0f : static_cast<float>(width) / static_cast<float>(height); }
};

inline SceneViewport ResolveSceneViewport(const SceneViewport& requested, uint32_t width, uint32_t height) {
    if (!requested.enabled) return {false, 0u, 0u, width, height};
    const uint32_t x = std::min(requested.x, width), y = std::min(requested.y, height);
    return {true, x, y, std::min(requested.width, width - x), std::min(requested.height, height - y)};
}

}  // namespace nuka::render
