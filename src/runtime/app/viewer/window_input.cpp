#include "runtime/app/viewer/window_input.hpp"
#include "runtime/app/viewer/camera_controller.hpp"

namespace nuka::runtime::app::viewer {
namespace {

ImGuiKey KeyFromKeysym(uint32_t symbol) {
    if (symbol >= 'a' && symbol <= 'z') return static_cast<ImGuiKey>(ImGuiKey_A + symbol - 'a');
    if (symbol >= 'A' && symbol <= 'Z') return static_cast<ImGuiKey>(ImGuiKey_A + symbol - 'A');
    if (symbol >= '0' && symbol <= '9') return static_cast<ImGuiKey>(ImGuiKey_0 + symbol - '0');
    if (symbol >= 0xffbeu && symbol <= 0xffd5u) return static_cast<ImGuiKey>(ImGuiKey_F1 + symbol - 0xffbeu);
    if (symbol >= 0xffb0u && symbol <= 0xffb9u) return static_cast<ImGuiKey>(ImGuiKey_Keypad0 + symbol - 0xffb0u);
    switch (symbol) {
        case 0xff09u: case 0xfe20u: return ImGuiKey_Tab;
        case 0xff51u: return ImGuiKey_LeftArrow;
        case 0xff53u: return ImGuiKey_RightArrow;
        case 0xff52u: return ImGuiKey_UpArrow;
        case 0xff54u: return ImGuiKey_DownArrow;
        case 0xff55u: return ImGuiKey_PageUp;
        case 0xff56u: return ImGuiKey_PageDown;
        case 0xff50u: return ImGuiKey_Home;
        case 0xff57u: return ImGuiKey_End;
        case 0xff63u: return ImGuiKey_Insert;
        case 0xffffu: return ImGuiKey_Delete;
        case 0xff08u: return ImGuiKey_Backspace;
        case 0xff0du: return ImGuiKey_Enter;
        case 0xff1bu: return ImGuiKey_Escape;
        case 0xffe5u: return ImGuiKey_CapsLock;
        case 0xff14u: return ImGuiKey_ScrollLock;
        case 0xff7fu: return ImGuiKey_NumLock;
        case 0xff61u: return ImGuiKey_PrintScreen;
        case 0xff13u: return ImGuiKey_Pause;
        case 0xffe1u: return ImGuiKey_LeftShift;
        case 0xffe2u: return ImGuiKey_RightShift;
        case 0xffe3u: return ImGuiKey_LeftCtrl;
        case 0xffe4u: return ImGuiKey_RightCtrl;
        case 0xffe9u: return ImGuiKey_LeftAlt;
        case 0xffeau: return ImGuiKey_RightAlt;
        case 0xffebu: return ImGuiKey_LeftSuper;
        case 0xffecu: return ImGuiKey_RightSuper;
        case 0xff67u: return ImGuiKey_Menu;
        case 0xff9eu: return ImGuiKey_Keypad0;
        case 0xff9cu: return ImGuiKey_Keypad1;
        case 0xff99u: return ImGuiKey_Keypad2;
        case 0xff9bu: return ImGuiKey_Keypad3;
        case 0xff96u: return ImGuiKey_Keypad4;
        case 0xff9du: return ImGuiKey_Keypad5;
        case 0xff98u: return ImGuiKey_Keypad6;
        case 0xff95u: return ImGuiKey_Keypad7;
        case 0xff97u: return ImGuiKey_Keypad8;
        case 0xff9au: return ImGuiKey_Keypad9;
        case 0xff9fu: case 0xffaeu: return ImGuiKey_KeypadDecimal;
        case 0xffafu: return ImGuiKey_KeypadDivide;
        case 0xffaau: return ImGuiKey_KeypadMultiply;
        case 0xffabu: return ImGuiKey_KeypadAdd;
        case 0xffadu: return ImGuiKey_KeypadSubtract;
        case 0xff8du: return ImGuiKey_KeypadEnter;
        case 0xffbdu: return ImGuiKey_KeypadEqual;
        case ' ': return ImGuiKey_Space;
        case ';': return ImGuiKey_Semicolon;
        case '\'': return ImGuiKey_Apostrophe;
        case ',': return ImGuiKey_Comma;
        case '.': return ImGuiKey_Period;
        case '/': return ImGuiKey_Slash;
        case '\\': return ImGuiKey_Backslash;
        case '[': return ImGuiKey_LeftBracket;
        case ']': return ImGuiKey_RightBracket;
        case '`': return ImGuiKey_GraveAccent;
        case '-': return ImGuiKey_Minus;
        case '=': return ImGuiKey_Equal;
        default: return ImGuiKey_None;
    }
}

}  // namespace

void WindowInput::Feed(const render::window::WindowEvent& event) {
    using Type = render::window::WindowEvent::Type;
    ImGuiIO& io = ImGui::GetIO();
    switch (event.type) {
        case Type::MouseMove:
            io.AddMousePosEvent(static_cast<float>(event.mouse_x), static_cast<float>(event.mouse_y));
            break;
        case Type::MouseButton:
            if (event.button <= 2u) io.AddMouseButtonEvent(event.button == 0u ? 0 : event.button == 1u ? 2 : 1, event.pressed);
            break;
        case Type::Scroll:
            io.AddMouseWheelEvent(0.0f, static_cast<float>(event.scroll_delta));
            break;
        case Type::Key: {
            ImGuiKey key = KeyFromKeysym(event.keysym);
            if (event.key < keys_by_code_.size()) {
                const bool layout = event.keysym == 0xfe03u || event.keysym == 0xff7eu || event.keysym == 0xfe11u;
                layout_keys_[event.key] = event.pressed && (layout || layout_keys_[event.key]);
                if (keys_by_code_[event.key] != ImGuiKey_None) key = keys_by_code_[event.key];
                keys_by_code_[event.key] = event.pressed ? key : ImGuiKey_None;
            }
            if (key == ImGuiKey_None) break;
            down_[key] = event.pressed;
            io.AddKeyEvent(ImGuiMod_Ctrl, down_[ImGuiKey_LeftCtrl] || down_[ImGuiKey_RightCtrl]);
            io.AddKeyEvent(ImGuiMod_Shift, down_[ImGuiKey_LeftShift] || down_[ImGuiKey_RightShift]);
            io.AddKeyEvent(ImGuiMod_Alt, down_[ImGuiKey_LeftAlt] || down_[ImGuiKey_RightAlt]);
            io.AddKeyEvent(ImGuiMod_Super, down_[ImGuiKey_LeftSuper] || down_[ImGuiKey_RightSuper]);
            io.AddKeyEvent(key, event.pressed);
            io.SetKeyEventNativeData(key, static_cast<int>(event.keysym), static_cast<int>(event.key));
            break;
        }
        case Type::TextInput:
            io.AddInputCharactersUTF8(event.text.c_str());
            break;
        case Type::FocusLost:
            for (int key = ImGuiKey_NamedKey_BEGIN; key < ImGuiKey_COUNT; ++key) {
                if (down_[static_cast<size_t>(key)]) io.AddKeyEvent(static_cast<ImGuiKey>(key), false);
            }
            io.AddKeyEvent(ImGuiMod_Ctrl, false);
            io.AddKeyEvent(ImGuiMod_Shift, false);
            io.AddKeyEvent(ImGuiMod_Alt, false);
            io.AddKeyEvent(ImGuiMod_Super, false);
            for (int button = 0; button < 5; ++button) io.AddMouseButtonEvent(button, false);
            keys_by_code_.fill(ImGuiKey_None);
            layout_keys_.fill(false);
            down_.fill(false);
            io.AddFocusEvent(false);
            break;
        case Type::FocusGained:
            io.AddFocusEvent(true);
            break;
        default: break;
    }
}

void ApplyCameraShortcuts(const render::RenderWorld& world, CameraController& camera,
                          scene::EntityId selected, bool viewport_hovered, bool gizmo_active, float aspect) {
    const ImGuiIO& io = ImGui::GetIO();
    if (!viewport_hovered || gizmo_active || io.AppFocusLost || io.WantCaptureKeyboard ||
        io.WantTextInput || io.KeyCtrl || io.KeyAlt || io.KeySuper || io.KeyShift) return;
    if (ImGui::IsKeyPressed(ImGuiKey_F, false)) camera.FrameSelected(world, selected, aspect);
    if (ImGui::IsKeyPressed(ImGuiKey_Home, false)) camera.FrameAll(world, aspect);
}

}  // namespace nuka::runtime::app::viewer
