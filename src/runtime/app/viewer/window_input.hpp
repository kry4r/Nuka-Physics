#pragma once

#include "render/window/window_surface.hpp"
#include "render/render_world.hpp"
#include "imgui.h"

#include <array>

namespace nuka::runtime::app::viewer {

class CameraController;

void ApplyCameraShortcuts(const render::RenderWorld& world, CameraController& camera,
                          scene::EntityId selected, bool viewport_hovered, bool gizmo_active);

class WindowInput {
public:
    void Feed(const render::window::WindowEvent& event);
    bool LayoutModifierDown() const {
        for (bool down : layout_keys_) if (down) return true;
        return false;
    }

private:
    std::array<ImGuiKey, 256> keys_by_code_{};
    std::array<bool, 256> layout_keys_{};
    std::array<bool, ImGuiKey_COUNT> down_{};
};

}  // namespace nuka::runtime::app::viewer
