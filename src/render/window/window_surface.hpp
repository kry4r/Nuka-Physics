#pragma once
// Window, input and Vulkan-surface seam of the present path: SDL3 on every desktop OS,
// VK_EXT_headless_surface without a display; it owns only the window and its VkSurfaceKHR.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// VkSurfaceKHR crosses this seam as void* (64-bit builds only) to keep Vulkan headers out.
struct VkInstance_T;

namespace nuka::render::window {

using WindowVkInstance = VkInstance_T*;
using WindowVkSurface  = void*;  // reinterpret to VkSurfaceKHR in the .cpp

// Modifier bits stamped on every WindowEvent with the state after that event.
// Layout covers AltGr / level-5 shifts, which type characters rather than shortcuts.
namespace modifier {
inline constexpr uint8_t kShift  = 1u << 0;
inline constexpr uint8_t kCtrl   = 1u << 1;
inline constexpr uint8_t kAlt    = 1u << 2;
inline constexpr uint8_t kSuper  = 1u << 3;
inline constexpr uint8_t kLayout = 1u << 4;
}  // namespace modifier

// One platform event in window coordinates (logical points; multiply by the
// framebuffer scale for pixels). ImGui receives the native event separately.
struct WindowEvent {
    enum class Type : uint8_t {
        None,
        Close,          // window manager close / quit request
        Resize,         // width/height carry the new framebuffer size in pixels
        MouseMove,      // mouse_x/mouse_y carry the new position
        MouseButton,    // button + pressed
        Key,            // key (USB HID usage) + pressed
        Scroll,         // scroll_delta carries wheel ticks (+up / -down)
        FocusLost,      // releases may be lost; clear latched input
        FocusGained,
        TextInput,      // committed UTF-8 text, separate from key bindings
    };
    Type     type = Type::None;
    uint32_t width = 0;
    uint32_t height = 0;
    int32_t  mouse_x = 0;
    int32_t  mouse_y = 0;
    uint32_t button = 0;     // 0=left, 1=middle, 2=right
    uint32_t key = 0;        // USB HID keyboard usage id (layout-independent physical key)
    uint8_t  modifiers = 0;  // modifier:: bits held after this event
    bool     pressed = false;
    int32_t  scroll_delta = 0;
    std::string text;

    bool Held(uint8_t bits) const { return (modifiers & bits) != 0u; }
};

// A platform window plus the VkSurfaceKHR it creates; the backend destroys the surface
// in its destructor, so the present renderer must release its swapchain first.
class WindowSurface {
public:
    virtual ~WindowSurface() = default;

    virtual WindowVkSurface CreateSurface(WindowVkInstance instance) = 0;
    virtual void            PollEvents(std::vector<WindowEvent>& out) = 0;
    virtual uint32_t        Width() const = 0;
    virtual uint32_t        Height() const = 0;
    virtual const char*     BackendName() const = 0;

    // SDL_Window* for binding imgui_impl_sdl3; nullptr on the headless surface.
    virtual void* NativeWindowHandle() const { return nullptr; }

    // Receives each native event (const SDL_Event*) before it is translated.
    virtual void SetNativeEventHook(std::function<void(const void*)> hook) { (void)hook; }

    // Instance extensions CreateSurface needs; constant for the backend's lifetime.
    virtual std::vector<std::string> RequiredInstanceExtensions() const = 0;
};

enum class SurfaceBackendKind : uint8_t {
    None,       // no backend could be created
    Sdl3,       // SDL3 window + VkSurfaceKHR (Windows, X11, Wayland)
    Headless,   // VK_EXT_headless_surface without a display
};

// Opens an SDL3 window when a video driver is available, else a headless surface
// when the loader advertises VK_EXT_headless_surface, else returns nullptr.
std::unique_ptr<WindowSurface> MakeSurface(const std::string& title,
                                           uint32_t width, uint32_t height,
                                           SurfaceBackendKind* out_kind = nullptr);

// A display-less surface of a fixed size, or nullptr when the extension is absent.
std::unique_ptr<WindowSurface> MakeHeadlessSurface(uint32_t width, uint32_t height);

}  // namespace nuka::render::window
