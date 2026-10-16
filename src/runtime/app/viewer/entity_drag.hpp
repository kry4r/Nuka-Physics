#pragma once

#include "render/window/window_surface.hpp"

#include <cstdint>

namespace nuka::runtime::app::viewer {

struct EntityDrag {
    static constexpr uint32_t kNone = ~uint32_t(0);
    uint32_t instance = kNone;

    void Cancel() { instance = kNone; }

    void Observe(const render::window::WindowEvent& event) {
        using Type = render::window::WindowEvent::Type;
        const bool release = event.type == Type::MouseButton && event.button == 0u && !event.pressed;
        const bool ctrl_release = event.type == Type::Key && !event.pressed &&
                                  (event.keysym == 0xffe3u || event.keysym == 0xffe4u);
        if (release || ctrl_release || event.type == Type::FocusLost || event.type == Type::Close) {
            Cancel();
        }
    }
};

}  // namespace nuka::runtime::app::viewer
