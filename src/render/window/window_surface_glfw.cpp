// ---------------------------------------------------------------------------
// nuka::render::window -- GLFW window + Vulkan-surface backend (the Windows
// present path; owner decision: GLFW + imgui_impl_glfw on Windows, xcb on Linux).
//
// Implements the same WindowSurface vtable the xcb backend does: open a GLFW
// window with no client API, create a VkSurfaceKHR via glfwCreateWindowSurface,
// drain glfw callbacks into the flat WindowEvent stream, and report the
// glfwGetRequiredInstanceExtensions set. The viewer additionally binds
// imgui_impl_glfw to NativeWindowHandle() so ImGui gets keyboard/char input.
//
// Built ONLY on Windows (the CMake nuka_present source branch); the whole TU is
// guarded so a stray non-Windows compile is an empty unit. HOST-ONLY / zero-CUDA.
// ---------------------------------------------------------------------------

#include "render/window/window_surface.hpp"

#ifdef _WIN32

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdio>

namespace nuka::render::window {

namespace {

// X11 keysym values for the modifier the viewer's picker tracks (kept identical
// to the xcb backend so the cross-platform viewer logic is unchanged).
constexpr uint32_t kKeysymCtrlL = 0xffe3u;
constexpr uint32_t kKeysymCtrlR = 0xffe4u;

void GlfwErrorCallback(int code, const char* description) {
    std::fprintf(stderr, "[nuka_window_glfw] GLFW error %d: %s\n", code,
                 description ? description : "(null)");
}

// ---------------------------------------------------------------------------
// GlfwWindowSurface -- a real GLFW window + VkSurfaceKHR (the Windows live path).
// ---------------------------------------------------------------------------
class GlfwWindowSurface final : public WindowSurface {
public:
    GlfwWindowSurface(const std::string& title, uint32_t width, uint32_t height)
        : width_(width), height_(height) {
        glfwSetErrorCallback(GlfwErrorCallback);
        if (glfwInit() != GLFW_TRUE) return;  // valid_ stays false
        glfw_owned_ = true;
        if (glfwVulkanSupported() != GLFW_TRUE) return;

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // Vulkan, no GL context
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
        window_ = glfwCreateWindow(static_cast<int>(width_), static_cast<int>(height_),
                                   title.c_str(), nullptr, nullptr);
        if (window_ == nullptr) return;

        // Seed the cached size from the actual framebuffer (pixels -> swapchain).
        int fb_w = 0, fb_h = 0;
        glfwGetFramebufferSize(window_, &fb_w, &fb_h);
        if (fb_w > 0 && fb_h > 0) {
            width_ = static_cast<uint32_t>(fb_w);
            height_ = static_cast<uint32_t>(fb_h);
        }

        glfwSetWindowUserPointer(window_, this);
        glfwSetKeyCallback(window_, &GlfwWindowSurface::KeyCb);
        glfwSetMouseButtonCallback(window_, &GlfwWindowSurface::MouseButtonCb);
        glfwSetCursorPosCallback(window_, &GlfwWindowSurface::CursorPosCb);
        glfwSetScrollCallback(window_, &GlfwWindowSurface::ScrollCb);
        glfwSetFramebufferSizeCallback(window_, &GlfwWindowSurface::FramebufferSizeCb);
        glfwSetWindowCloseCallback(window_, &GlfwWindowSurface::WindowCloseCb);
        glfwSetWindowFocusCallback(window_, &GlfwWindowSurface::WindowFocusCb);
        valid_ = true;
    }

    ~GlfwWindowSurface() override {
        if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
        }
        if (window_ != nullptr) glfwDestroyWindow(window_);
        if (glfw_owned_) glfwTerminate();
    }

    bool valid() const { return valid_; }

    WindowVkSurface CreateSurface(WindowVkInstance instance) override {
        if (!valid_) return nullptr;
        if (surface_ != VK_NULL_HANDLE) {
            return reinterpret_cast<WindowVkSurface>(surface_);
        }
        instance_ = reinterpret_cast<VkInstance>(instance);
        if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS) {
            surface_ = VK_NULL_HANDLE;
            return nullptr;
        }
        return reinterpret_cast<WindowVkSurface>(surface_);
    }

    void PollEvents(std::vector<WindowEvent>& out) override {
        if (!valid_) return;
        glfwPollEvents();  // fires the callbacks below, which append to queue_
        if (queue_.empty()) return;
        out.insert(out.end(), queue_.begin(), queue_.end());
        queue_.clear();
    }

    uint32_t Width() const override { return width_; }
    uint32_t Height() const override { return height_; }
    const char* BackendName() const override { return "glfw"; }
    void* NativeWindowHandle() const override { return window_; }

    std::vector<std::string> RequiredInstanceExtensions() const override {
        std::vector<std::string> exts{"VK_KHR_surface"};
        uint32_t count = 0;
        const char** names = glfwGetRequiredInstanceExtensions(&count);
        for (uint32_t i = 0; names != nullptr && i < count; ++i) {
            const std::string name = names[i];
            if (name != "VK_KHR_surface") exts.push_back(name);
        }
        return exts;
    }

private:
    static GlfwWindowSurface* Self(GLFWwindow* w) {
        return static_cast<GlfwWindowSurface*>(glfwGetWindowUserPointer(w));
    }

    static void KeyCb(GLFWwindow* w, int key, int /*scancode*/, int action, int /*mods*/) {
        GlfwWindowSurface* self = Self(w);
        if (self == nullptr) return;
        WindowEvent ev;
        ev.type = WindowEvent::Type::Key;
        ev.key = static_cast<uint32_t>(key);
        if (key == GLFW_KEY_LEFT_CONTROL) ev.keysym = kKeysymCtrlL;
        else if (key == GLFW_KEY_RIGHT_CONTROL) ev.keysym = kKeysymCtrlR;
        ev.pressed = (action != GLFW_RELEASE);
        self->queue_.push_back(ev);
    }

    static void MouseButtonCb(GLFWwindow* w, int button, int action, int /*mods*/) {
        GlfwWindowSurface* self = Self(w);
        if (self == nullptr) return;
        WindowEvent ev;
        ev.type = WindowEvent::Type::MouseButton;
        // WindowEvent button order is xcb-style: 0=left, 1=middle, 2=right.
        ev.button = (button == GLFW_MOUSE_BUTTON_LEFT)
                        ? 0u
                        : (button == GLFW_MOUSE_BUTTON_MIDDLE) ? 1u : 2u;
        ev.pressed = (action != GLFW_RELEASE);
        ev.mouse_x = self->last_x_;
        ev.mouse_y = self->last_y_;
        self->queue_.push_back(ev);
    }

    static void CursorPosCb(GLFWwindow* w, double x, double y) {
        GlfwWindowSurface* self = Self(w);
        if (self == nullptr) return;
        self->last_x_ = static_cast<int32_t>(x);
        self->last_y_ = static_cast<int32_t>(y);
        WindowEvent ev;
        ev.type = WindowEvent::Type::MouseMove;
        ev.mouse_x = self->last_x_;
        ev.mouse_y = self->last_y_;
        self->queue_.push_back(ev);
    }

    static void ScrollCb(GLFWwindow* w, double /*xoff*/, double yoff) {
        GlfwWindowSurface* self = Self(w);
        if (self == nullptr) return;
        if (yoff == 0.0) return;
        WindowEvent ev;
        ev.type = WindowEvent::Type::Scroll;
        ev.scroll_delta = (yoff > 0.0) ? 1 : -1;  // +up / -down, like the xcb wheel
        ev.mouse_x = self->last_x_;
        ev.mouse_y = self->last_y_;
        self->queue_.push_back(ev);
    }

    static void FramebufferSizeCb(GLFWwindow* w, int width, int height) {
        GlfwWindowSurface* self = Self(w);
        if (self == nullptr || width <= 0 || height <= 0) return;
        self->width_ = static_cast<uint32_t>(width);
        self->height_ = static_cast<uint32_t>(height);
        WindowEvent ev;
        ev.type = WindowEvent::Type::Resize;
        ev.width = self->width_;
        ev.height = self->height_;
        self->queue_.push_back(ev);
    }

    static void WindowCloseCb(GLFWwindow* w) {
        GlfwWindowSurface* self = Self(w);
        if (self == nullptr) return;
        WindowEvent ev;
        ev.type = WindowEvent::Type::Close;
        self->queue_.push_back(ev);
    }

    static void WindowFocusCb(GLFWwindow* w, int focused) {
        GlfwWindowSurface* self = Self(w);
        if (self == nullptr || focused != 0) return;
        // Focus left: releases are lost; consumers reset latched key state.
        WindowEvent ev;
        ev.type = WindowEvent::Type::FocusLost;
        self->queue_.push_back(ev);
    }

    GLFWwindow*  window_   = nullptr;
    VkInstance   instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_  = VK_NULL_HANDLE;
    std::vector<WindowEvent> queue_;
    int32_t  last_x_ = 0;
    int32_t  last_y_ = 0;
    uint32_t width_  = 0;
    uint32_t height_ = 0;
    bool     glfw_owned_ = false;
    bool     valid_      = false;
};

}  // namespace

std::unique_ptr<WindowSurface> MakeSurface(const std::string& title,
                                           uint32_t width, uint32_t height,
                                           SurfaceBackendKind* out_kind) {
    auto set_kind = [&](SurfaceBackendKind k) {
        if (out_kind != nullptr) *out_kind = k;
    };
    auto glfw_surface = std::make_unique<GlfwWindowSurface>(title, width, height);
    if (glfw_surface->valid()) {
        set_kind(SurfaceBackendKind::Glfw);
        return glfw_surface;
    }
    set_kind(SurfaceBackendKind::None);
    return nullptr;
}

std::vector<std::string> PreferredSurfaceInstanceExtensions(SurfaceBackendKind* out_kind) {
    auto set_kind = [&](SurfaceBackendKind k) {
        if (out_kind != nullptr) *out_kind = k;
    };
    if (glfwInit() != GLFW_TRUE || glfwVulkanSupported() != GLFW_TRUE) {
        set_kind(SurfaceBackendKind::None);
        return {};
    }
    std::vector<std::string> exts{"VK_KHR_surface"};
    uint32_t count = 0;
    const char** names = glfwGetRequiredInstanceExtensions(&count);
    for (uint32_t i = 0; names != nullptr && i < count; ++i) {
        const std::string name = names[i];
        if (name != "VK_KHR_surface") exts.push_back(name);
    }
    set_kind(SurfaceBackendKind::Glfw);
    return exts;
}

}  // namespace nuka::render::window

#endif  // _WIN32
