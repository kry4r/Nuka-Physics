#pragma once
// ---------------------------------------------------------------------------
// nuka::render::window -- the thin self-written windowing / Vulkan-surface
// backend (self-written xcb on Linux; GLFW on Windows).
//
// This is the SURFACE SOURCE for the swapchain present path
// (vulkan_present_renderer). It is deliberately tiny: create a window, create a
// VkSurfaceKHR on it, pump platform events into a small POD event stream the
// viewer (T3) consumes, and tear down. It owns NO Vulkan device state -- only the
// platform window + the VkSurfaceKHR it created against a caller-supplied
// VkInstance. The present renderer creates the surface FIRST (so the renderer's
// device selection can verify graphics+present on it), then builds its swapchain.
//
// RUNTIME-SELECTABLE backend (recon 1.3): MakeSurface() prefers
// VK_EXT_headless_surface (vkCreateHeadlessSurfaceEXT) when the loader exports it
// (a modern / CI box), ELSE falls back to the xcb backend (this box, runs under
// Xvfb). Detection is at runtime (instance-ext query + a loader dlsym of the
// headless entry point), so the SAME binary verifies on both. On the 1.2.131
// loader here the headless path is unavailable -> xcb is the live path.
//
// HOST-ONLY / zero-CUDA-token: pure C++ / Vulkan / xcb. No CUDA of any kind.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Vulkan handles at the boundary. We forward-declare the few we name so this
// header stays light; the .cpp includes <vulkan/vulkan.h>. VkInstance is a
// dispatchable (pointer) handle; VkSurfaceKHR is non-dispatchable (64-bit) which
// we carry as a void* at this seam (a pointer on the 64-bit builds we target).
struct VkInstance_T;

namespace nuka::render::window {

using WindowVkInstance = VkInstance_T*;
using WindowVkSurface  = void*;  // reinterpret to VkSurfaceKHR in the .cpp

// ---------------------------------------------------------------------------
// WindowEvent -- the minimal-but-real input the viewer consumes. One flat
// POD per platform event; the viewer maps these onto the CommandQueue (play/
// pause/camera) + ImGui io. Kept small on purpose (this is a showcase viewer,
// not a game-engine input system).
// ---------------------------------------------------------------------------
struct WindowEvent {
    enum class Type : uint8_t {
        None,
        Close,          // window manager close / destroy
        Resize,         // width/height carry the new client size
        MouseMove,      // mouse_x/mouse_y carry the new position
        MouseButton,    // button + pressed
        Key,            // key (raw keycode) + keysym (resolved) + pressed
        Scroll,         // scroll_delta carries wheel ticks (+up / -down)
        FocusLost,      // keyboard focus left the window (releases are lost:
                        // consumers must reset any latched modifier/drag state)
    };
    Type     type = Type::None;
    uint32_t width = 0;
    uint32_t height = 0;
    int32_t  mouse_x = 0;
    int32_t  mouse_y = 0;
    uint32_t button = 0;     // 0=left, 1=middle, 2=right (xcb button order)
    uint32_t key = 0;        // RAW platform keycode (evdev hardware code; keymap-dependent)
    uint32_t keysym = 0;     // resolved X11 keysym (XK_*/XKB_KEY_*); keymap-independent, 0 if unresolved
    bool     pressed = false;
    int32_t  scroll_delta = 0;
};

// ---------------------------------------------------------------------------
// WindowSurface -- the abstract backend interface.
//
//   * CreateSurface(instance): create the VkSurfaceKHR for this window against
//     `instance` (which MUST have been created with the matching surface
//     instance extensions). Returns the opaque surface handle (or nullptr on
//     failure). Idempotent: returns the cached handle on repeat calls.
//   * PollEvents(): drain the platform queue into `out`, appending zero or more
//     WindowEvents. Non-blocking. Updates cached width/height on resize.
//   * Width()/Height(): the current client size.
//   * BackendName(): "xcb" or "headless" for the report.
//
// The surface is destroyed in the backend dtor (it owns it), AFTER the present
// renderer has torn down its swapchain (caller ordering responsibility).
// ---------------------------------------------------------------------------
class WindowSurface {
public:
    virtual ~WindowSurface() = default;

    virtual WindowVkSurface CreateSurface(WindowVkInstance instance) = 0;
    virtual void            PollEvents(std::vector<WindowEvent>& out) = 0;
    virtual uint32_t        Width() const = 0;
    virtual uint32_t        Height() const = 0;
    virtual const char*     BackendName() const = 0;

    // The native platform window handle (GLFWwindow* on the GLFW backend) so a
    // caller can bind a platform ImGui backend (imgui_impl_glfw). nullptr when the
    // backend has no such handle (xcb / headless), which keep self-fed input.
    virtual void* NativeWindowHandle() const { return nullptr; }

    // The instance extensions THIS backend needs enabled on the VkInstance for
    // CreateSurface to succeed (e.g. {"VK_KHR_surface","VK_KHR_xcb_surface"}).
    // The caller enables these on its instance before CreateSurface. Static-ish:
    // a backend reports the same set for its lifetime.
    virtual std::vector<std::string> RequiredInstanceExtensions() const = 0;
};

// ---------------------------------------------------------------------------
// SurfaceBackendKind -- which concrete backend MakeSurface produced.
// ---------------------------------------------------------------------------
enum class SurfaceBackendKind : uint8_t {
    None,       // no backend could be created (no display / no surface ext)
    Xcb,        // self-written xcb window (the live path here, under Xvfb)
    Headless,   // VK_EXT_headless_surface (modern loader / CI only)
    Glfw,       // GLFW window + VkSurfaceKHR (the Windows present path)
};

// ---------------------------------------------------------------------------
// MakeSurface -- the runtime-selectable factory.
//
// Tries, in order:
//   1. Headless (VK_EXT_headless_surface) IF the loader exports
//      vkCreateHeadlessSurfaceEXT AND the instance ext is advertised -- a modern
//      / CI box. (Returns VK_ERROR_EXTENSION_NOT_PRESENT here -> skipped.)
//   2. Xcb: open the X display ($DISPLAY, e.g. Xvfb's :99), create a window of
//      the requested size, ready for vkCreateXcbSurfaceKHR.
//
// Returns nullptr (and sets *out_kind = None) if neither is available -- the
// caller (the present smoke) then GTEST_SKIPs cleanly. `title`/`width`/`height`
// seed the window. NOTE: this does NOT need a VkInstance yet (it only opens the
// platform window); CreateSurface() takes the instance.
// ---------------------------------------------------------------------------
std::unique_ptr<WindowSurface> MakeSurface(const std::string& title,
                                           uint32_t width, uint32_t height,
                                           SurfaceBackendKind* out_kind = nullptr);

// The instance extensions the CURRENTLY-SELECTABLE backend would need, WITHOUT
// creating a window -- so the present renderer can decide whether to construct in
// present mode. Returns the xcb set when a display is present, the headless set
// when only that is available, else empty. (A convenience for callers that want
// to enable the right ext set before opening the window; the present renderer
// just enables surface+xcb+headless opportunistically, so this is informational.)
std::vector<std::string> PreferredSurfaceInstanceExtensions(SurfaceBackendKind* out_kind = nullptr);

}  // namespace nuka::render::window
