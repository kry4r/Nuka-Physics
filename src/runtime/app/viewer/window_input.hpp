#pragma once

#include "render/render_world.hpp"

namespace nuka::runtime::app::viewer {

class CameraController;

// F frames the selection and Home frames everything while the pointer is over the viewport.
void ApplyCameraShortcuts(const render::RenderWorld& world, CameraController& camera,
                          scene::EntityId selected, bool viewport_hovered, bool gizmo_active, float aspect = 1.0f);

}  // namespace nuka::runtime::app::viewer
