// SDL3 window backend: one resizable high-density Vulkan window whose native events go
// to the hook (imgui_impl_sdl3) and, translated, to the WindowEvent stream.

#include "render/window/window_surface.hpp"

#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <cmath>
#include <cstdio>
#include <utility>

namespace nuka::render::window {

namespace {

constexpr SDL_InitFlags kSubsystems = SDL_INIT_VIDEO | SDL_INIT_GAMEPAD;

uint8_t ModifierBits(SDL_Keymod mod) {
    uint8_t bits = 0u;
    if (mod & SDL_KMOD_SHIFT) bits |= modifier::kShift;
    if (mod & SDL_KMOD_CTRL) bits |= modifier::kCtrl;
    if (mod & SDL_KMOD_ALT) bits |= modifier::kAlt;
    if (mod & SDL_KMOD_GUI) bits |= modifier::kSuper;
    if (mod & (SDL_KMOD_MODE | SDL_KMOD_LEVEL5)) bits |= modifier::kLayout;
    return bits;
}

int32_t Point(float value) { return static_cast<int32_t>(std::lround(value)); }

class Sdl3WindowSurface final : public WindowSurface {
public:
    Sdl3WindowSurface(const std::string& title, uint32_t width, uint32_t height) {
        if (!SDL_InitSubSystem(kSubsystems)) {
            std::fprintf(stderr, "[nuka_window] SDL video unavailable: %s\n", SDL_GetError());
            return;
        }
        sdl_owned_ = true;
        const SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
        window_ = SDL_CreateWindow(title.c_str(), static_cast<int>(width), static_cast<int>(height), flags);
        if (window_ == nullptr) {
            std::fprintf(stderr, "[nuka_window] SDL_CreateWindow failed: %s\n", SDL_GetError());
            return;
        }
        RefreshPixelSize();
        valid_ = true;
    }

    ~Sdl3WindowSurface() override {
        if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
            SDL_Vulkan_DestroySurface(instance_, surface_, nullptr);
        }
        if (window_ != nullptr) SDL_DestroyWindow(window_);
        if (sdl_owned_) SDL_QuitSubSystem(kSubsystems);
    }

    bool valid() const { return valid_; }

    WindowVkSurface CreateSurface(WindowVkInstance instance) override {
        if (surface_ != VK_NULL_HANDLE) return reinterpret_cast<WindowVkSurface>(surface_);
        instance_ = reinterpret_cast<VkInstance>(instance);
        if (!SDL_Vulkan_CreateSurface(window_, instance_, nullptr, &surface_)) {
            std::fprintf(stderr, "[nuka_window] SDL_Vulkan_CreateSurface failed: %s\n", SDL_GetError());
            surface_ = VK_NULL_HANDLE;
            return nullptr;
        }
        return reinterpret_cast<WindowVkSurface>(surface_);
    }

    void PollEvents(std::vector<WindowEvent>& out) override {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (hook_) hook_(&event);
            Translate(event, out);
        }
    }

    uint32_t Width() const override { return width_; }
    uint32_t Height() const override { return height_; }
    const char* BackendName() const override { return "sdl3"; }
    void* NativeWindowHandle() const override { return window_; }
    void SetNativeEventHook(std::function<void(const void*)> hook) override { hook_ = std::move(hook); }

    std::vector<std::string> RequiredInstanceExtensions() const override {
        Uint32 count = 0;
        const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);
        std::vector<std::string> exts;
        for (Uint32 i = 0; names != nullptr && i < count; ++i) exts.emplace_back(names[i]);
        return exts;
    }

private:
    void RefreshPixelSize() {
        int w = 0, h = 0;
        if (SDL_GetWindowSizeInPixels(window_, &w, &h) && w >= 0 && h >= 0) {
            width_ = static_cast<uint32_t>(w);
            height_ = static_cast<uint32_t>(h);
        }
    }

    void Emit(std::vector<WindowEvent>& out, WindowEvent ev) const {
        ev.modifiers = modifiers_;
        out.push_back(std::move(ev));
    }

    void EmitResize(std::vector<WindowEvent>& out) const {
        WindowEvent ev;
        ev.type = WindowEvent::Type::Resize;
        ev.width = width_;
        ev.height = height_;
        Emit(out, std::move(ev));
    }

    void Translate(const SDL_Event& e, std::vector<WindowEvent>& out) {
        WindowEvent ev;
        switch (e.type) {
            case SDL_EVENT_QUIT:
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                ev.type = WindowEvent::Type::Close;
                Emit(out, std::move(ev));
                break;
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            case SDL_EVENT_WINDOW_RESTORED:
            case SDL_EVENT_WINDOW_MAXIMIZED:
                RefreshPixelSize();
                EmitResize(out);
                break;
            case SDL_EVENT_WINDOW_MINIMIZED:
                width_ = height_ = 0u;
                EmitResize(out);
                break;
            case SDL_EVENT_WINDOW_FOCUS_GAINED:
                ev.type = WindowEvent::Type::FocusGained;
                Emit(out, std::move(ev));
                break;
            case SDL_EVENT_WINDOW_FOCUS_LOST:
                modifiers_ = 0u;
                ev.type = WindowEvent::Type::FocusLost;
                Emit(out, std::move(ev));
                break;
            case SDL_EVENT_MOUSE_MOTION:
                ev.type = WindowEvent::Type::MouseMove;
                ev.mouse_x = Point(e.motion.x);
                ev.mouse_y = Point(e.motion.y);
                Emit(out, std::move(ev));
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (e.button.button == SDL_BUTTON_LEFT) ev.button = 0u;
                else if (e.button.button == SDL_BUTTON_MIDDLE) ev.button = 1u;
                else if (e.button.button == SDL_BUTTON_RIGHT) ev.button = 2u;
                else break;
                ev.type = WindowEvent::Type::MouseButton;
                ev.pressed = e.button.down;
                ev.mouse_x = Point(e.button.x);
                ev.mouse_y = Point(e.button.y);
                Emit(out, std::move(ev));
                break;
            case SDL_EVENT_MOUSE_WHEEL: {
                const int32_t ticks = e.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -e.wheel.integer_y
                                                                                 : e.wheel.integer_y;
                if (ticks == 0) break;
                ev.type = WindowEvent::Type::Scroll;
                ev.scroll_delta = ticks;
                ev.mouse_x = Point(e.wheel.mouse_x);
                ev.mouse_y = Point(e.wheel.mouse_y);
                Emit(out, std::move(ev));
                break;
            }
            case SDL_EVENT_KEY_DOWN:
            case SDL_EVENT_KEY_UP:
                modifiers_ = ModifierBits(e.key.mod);
                if (e.key.repeat) break;
                ev.type = WindowEvent::Type::Key;
                ev.key = static_cast<uint32_t>(e.key.scancode);
                ev.pressed = e.key.down;
                Emit(out, std::move(ev));
                break;
            case SDL_EVENT_TEXT_INPUT:
                if (e.text.text == nullptr) break;
                ev.type = WindowEvent::Type::TextInput;
                ev.text = e.text.text;
                Emit(out, std::move(ev));
                break;
            default:
                break;
        }
    }

    SDL_Window*  window_   = nullptr;
    VkInstance   instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_  = VK_NULL_HANDLE;
    std::function<void(const void*)> hook_;
    uint32_t width_  = 0;
    uint32_t height_ = 0;
    uint8_t  modifiers_ = 0;
    bool     sdl_owned_ = false;
    bool     valid_     = false;
};

}  // namespace

std::unique_ptr<WindowSurface> MakeSurface(const std::string& title, uint32_t width, uint32_t height,
                                           SurfaceBackendKind* out_kind) {
    auto set_kind = [&](SurfaceBackendKind k) {
        if (out_kind != nullptr) *out_kind = k;
    };
    auto sdl = std::make_unique<Sdl3WindowSurface>(title, width, height);
    if (sdl->valid()) {
        set_kind(SurfaceBackendKind::Sdl3);
        return sdl;
    }
    sdl.reset();
    if (auto headless = MakeHeadlessSurface(width, height)) {
        set_kind(SurfaceBackendKind::Headless);
        return headless;
    }
    set_kind(SurfaceBackendKind::None);
    return nullptr;
}

}  // namespace nuka::render::window
