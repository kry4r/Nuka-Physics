#include "runtime/app/viewer/window_input.hpp"
#include "runtime/app/viewer/camera_controller.hpp"

#include "imgui.h"

namespace nuka::runtime::app::viewer {

void ApplyCameraShortcuts(const render::RenderWorld& world, CameraController& camera,
                          scene::EntityId selected, bool viewport_hovered, bool gizmo_active, float aspect) {
    const ImGuiIO& io = ImGui::GetIO();
    if (!viewport_hovered || gizmo_active || io.AppFocusLost || io.WantCaptureKeyboard ||
        io.WantTextInput || io.KeyCtrl || io.KeyAlt || io.KeySuper || io.KeyShift) return;
    if (ImGui::IsKeyPressed(ImGuiKey_F, false)) camera.FrameSelected(world, selected, aspect);
    if (ImGui::IsKeyPressed(ImGuiKey_Home, false)) camera.FrameAll(world, aspect);
}

}  // namespace nuka::runtime::app::viewer
