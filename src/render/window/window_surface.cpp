// ---------------------------------------------------------------------------
// nuka::render::window -- xcb + headless surface backend implementation (M8.5 T2).
//
// Self-written, thin (Decision D1). The xcb backend opens an X connection
// ($DISPLAY -- under Xvfb here), creates a window, and produces a VkSurfaceKHR via
// vkCreateXcbSurfaceKHR. The headless backend (VK_EXT_headless_surface) is created
// only when the loader exports vkCreateHeadlessSurfaceEXT (NOT on the 1.2.131
// loader here -- runtime-detected). Events are pumped from the xcb queue into the
// flat WindowEvent stream.
//
// HOST-ONLY / zero-CUDA-token: pure C++ / Vulkan / xcb. No CUDA tokens.
// ---------------------------------------------------------------------------

#include "render/window/window_surface.hpp"

#define VK_USE_PLATFORM_XCB_KHR
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <xcb/xcb.h>

#include <cstdlib>
#include <cstring>

namespace nuka::render::window {

namespace {

// Runtime check: does the loader export vkCreateHeadlessSurfaceEXT? On a modern
// loader this is a real entry point; on the 1.2.131 loader here it is absent, so
// the headless backend is never selected (recon 1.3). We dlsym the process image
// (the loader is already linked) -- no extra dependency.
bool LoaderExportsHeadlessSurface() {
    void* sym = dlsym(RTLD_DEFAULT, "vkCreateHeadlessSurfaceEXT");
    return sym != nullptr;
}

bool InstanceExtAdvertised(const char* name) {
    uint32_t count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS ||
        count == 0u) {
        return false;
    }
    std::vector<VkExtensionProperties> exts(count);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, exts.data()) != VK_SUCCESS) {
        return false;
    }
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
}

bool DisplayAvailable() {
    const char* display = std::getenv("DISPLAY");
    return display != nullptr && display[0] != '\0';
}

// ---------------------------------------------------------------------------
// XcbWindowSurface -- a real xcb window + VkSurfaceKHR (the live path under Xvfb).
// ---------------------------------------------------------------------------
class XcbWindowSurface final : public WindowSurface {
public:
    XcbWindowSurface(const std::string& title, uint32_t width, uint32_t height)
        : width_(width), height_(height) {
        int screen_index = 0;
        connection_ = xcb_connect(nullptr, &screen_index);
        if (connection_ == nullptr || xcb_connection_has_error(connection_) != 0) {
            if (connection_ != nullptr) {
                xcb_disconnect(connection_);
                connection_ = nullptr;
            }
            return;  // valid_ stays false
        }

        const xcb_setup_t* setup = xcb_get_setup(connection_);
        xcb_screen_iterator_t it = xcb_setup_roots_iterator(setup);
        for (int i = 0; i < screen_index && it.rem != 0; ++i) {
            xcb_screen_next(&it);
        }
        screen_ = it.data;
        if (screen_ == nullptr) {
            xcb_disconnect(connection_);
            connection_ = nullptr;
            return;
        }

        window_ = xcb_generate_id(connection_);
        const uint32_t mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
        const uint32_t values[2] = {
            screen_->black_pixel,
            XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_STRUCTURE_NOTIFY |
                XCB_EVENT_MASK_KEY_PRESS | XCB_EVENT_MASK_KEY_RELEASE |
                XCB_EVENT_MASK_BUTTON_PRESS | XCB_EVENT_MASK_BUTTON_RELEASE |
                XCB_EVENT_MASK_POINTER_MOTION,
        };
        xcb_create_window(connection_, XCB_COPY_FROM_PARENT, window_, screen_->root,
                          0, 0, static_cast<uint16_t>(width_), static_cast<uint16_t>(height_),
                          0, XCB_WINDOW_CLASS_INPUT_OUTPUT, screen_->root_visual,
                          mask, values);

        // Title (best-effort; ignored if it fails).
        xcb_change_property(connection_, XCB_PROP_MODE_REPLACE, window_,
                            XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
                            static_cast<uint32_t>(title.size()), title.c_str());

        // Register a WM_DELETE_WINDOW atom so the window manager close routes to a
        // ClientMessage we translate to a Close event.
        xcb_intern_atom_cookie_t proto_cookie =
            xcb_intern_atom(connection_, 1, 12, "WM_PROTOCOLS");
        xcb_intern_atom_reply_t* proto_reply =
            xcb_intern_atom_reply(connection_, proto_cookie, nullptr);
        xcb_intern_atom_cookie_t del_cookie =
            xcb_intern_atom(connection_, 0, 16, "WM_DELETE_WINDOW");
        xcb_intern_atom_reply_t* del_reply =
            xcb_intern_atom_reply(connection_, del_cookie, nullptr);
        if (proto_reply != nullptr && del_reply != nullptr) {
            delete_atom_ = del_reply->atom;
            xcb_change_property(connection_, XCB_PROP_MODE_REPLACE, window_,
                                proto_reply->atom, XCB_ATOM_ATOM, 32, 1, &del_reply->atom);
        }
        std::free(proto_reply);
        std::free(del_reply);

        xcb_map_window(connection_, window_);
        xcb_flush(connection_);
        FetchKeyboardMapping();  // keycode->keysym table (core xcb, no xcb-keysyms dep)
        valid_ = true;
    }

    ~XcbWindowSurface() override {
        std::free(keymap_reply_);
        if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
        }
        if (connection_ != nullptr) {
            if (window_ != 0u) xcb_destroy_window(connection_, window_);
            xcb_disconnect(connection_);
        }
    }

    bool valid() const { return valid_; }

    WindowVkSurface CreateSurface(WindowVkInstance instance) override {
        if (!valid_) return nullptr;
        if (surface_ != VK_NULL_HANDLE) {
            return reinterpret_cast<WindowVkSurface>(surface_);
        }
        instance_ = reinterpret_cast<VkInstance>(instance);
        VkXcbSurfaceCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
        info.connection = connection_;
        info.window = window_;
        if (vkCreateXcbSurfaceKHR(instance_, &info, nullptr, &surface_) != VK_SUCCESS) {
            surface_ = VK_NULL_HANDLE;
            return nullptr;
        }
        return reinterpret_cast<WindowVkSurface>(surface_);
    }

    void PollEvents(std::vector<WindowEvent>& out) override {
        if (!valid_) return;
        xcb_generic_event_t* event = nullptr;
        while ((event = xcb_poll_for_event(connection_)) != nullptr) {
            const uint8_t kind = static_cast<uint8_t>(event->response_type & ~0x80u);
            switch (kind) {
                case XCB_CONFIGURE_NOTIFY: {
                    const auto* cfg = reinterpret_cast<xcb_configure_notify_event_t*>(event);
                    if (cfg->width != 0 && cfg->height != 0 &&
                        (cfg->width != width_ || cfg->height != height_)) {
                        width_ = cfg->width;
                        height_ = cfg->height;
                        WindowEvent ev;
                        ev.type = WindowEvent::Type::Resize;
                        ev.width = width_;
                        ev.height = height_;
                        out.push_back(ev);
                    }
                    break;
                }
                case XCB_MOTION_NOTIFY: {
                    const auto* mv = reinterpret_cast<xcb_motion_notify_event_t*>(event);
                    WindowEvent ev;
                    ev.type = WindowEvent::Type::MouseMove;
                    ev.mouse_x = mv->event_x;
                    ev.mouse_y = mv->event_y;
                    out.push_back(ev);
                    break;
                }
                case XCB_BUTTON_PRESS:
                case XCB_BUTTON_RELEASE: {
                    const auto* bt = reinterpret_cast<xcb_button_press_event_t*>(event);
                    // xcb buttons: 1=left 2=middle 3=right; 4/5 = wheel up/down.
                    if (bt->detail == 4u || bt->detail == 5u) {
                        if (kind == XCB_BUTTON_PRESS) {
                            WindowEvent ev;
                            ev.type = WindowEvent::Type::Scroll;
                            ev.scroll_delta = (bt->detail == 4u) ? 1 : -1;
                            ev.mouse_x = bt->event_x;
                            ev.mouse_y = bt->event_y;
                            out.push_back(ev);
                        }
                    } else {
                        WindowEvent ev;
                        ev.type = WindowEvent::Type::MouseButton;
                        ev.button = (bt->detail == 1u) ? 0u : (bt->detail == 2u) ? 1u : 2u;
                        ev.pressed = (kind == XCB_BUTTON_PRESS);
                        ev.mouse_x = bt->event_x;
                        ev.mouse_y = bt->event_y;
                        out.push_back(ev);
                    }
                    break;
                }
                case XCB_KEY_PRESS:
                case XCB_KEY_RELEASE: {
                    const auto* kp = reinterpret_cast<xcb_key_press_event_t*>(event);
                    WindowEvent ev;
                    ev.type = WindowEvent::Type::Key;
                    ev.key = kp->detail;  // RAW keycode (kept for any keycode-level consumer)
                    ev.keysym = ResolveKeysym(kp->detail);  // keymap-independent (XK_*)
                    ev.pressed = (kind == XCB_KEY_PRESS);
                    out.push_back(ev);
                    break;
                }
                case XCB_CLIENT_MESSAGE: {
                    const auto* cm = reinterpret_cast<xcb_client_message_event_t*>(event);
                    if (delete_atom_ != 0u && cm->data.data32[0] == delete_atom_) {
                        WindowEvent ev;
                        ev.type = WindowEvent::Type::Close;
                        out.push_back(ev);
                    }
                    break;
                }
                case XCB_DESTROY_NOTIFY: {
                    WindowEvent ev;
                    ev.type = WindowEvent::Type::Close;
                    out.push_back(ev);
                    break;
                }
                default:
                    break;
            }
            std::free(event);
        }
    }

    uint32_t Width() const override { return width_; }
    uint32_t Height() const override { return height_; }
    const char* BackendName() const override { return "xcb"; }

    std::vector<std::string> RequiredInstanceExtensions() const override {
        return {"VK_KHR_surface", "VK_KHR_xcb_surface"};
    }

private:
    // Fetch the server's keycode->keysym table ONCE via the core X protocol
    // (xcb_get_keyboard_mapping). No xcb-keysyms / xkbcommon dependency needed.
    void FetchKeyboardMapping() {
        if (connection_ == nullptr) return;
        const xcb_setup_t* setup = xcb_get_setup(connection_);
        if (setup == nullptr) return;
        keycode_min_ = setup->min_keycode;
        const uint8_t keycode_max = setup->max_keycode;
        if (keycode_max < keycode_min_) return;
        // count is a u8 protocol field; (max-min+1) is <=255 for a valid range.
        const uint32_t span = static_cast<uint32_t>(keycode_max - keycode_min_) + 1u;
        const uint8_t count = static_cast<uint8_t>(span > 255u ? 255u : span);
        xcb_get_keyboard_mapping_cookie_t cookie =
            xcb_get_keyboard_mapping(connection_, keycode_min_, count);
        keymap_reply_ = xcb_get_keyboard_mapping_reply(connection_, cookie, nullptr);
        if (keymap_reply_ != nullptr) {
            keymap_keysyms_ = xcb_get_keyboard_mapping_keysyms(keymap_reply_);
            keysyms_len_ =
                static_cast<uint32_t>(xcb_get_keyboard_mapping_keysyms_length(keymap_reply_));
        }
    }

    // Resolve a raw keycode to its first (unshifted, group 0) keysym, or 0. The
    // keysyms are laid out keysyms_per_keycode per code from keycode_min_.
    uint32_t ResolveKeysym(uint8_t keycode) const {
        if (keymap_reply_ == nullptr || keymap_keysyms_ == nullptr) return 0u;
        if (keycode < keycode_min_) return 0u;
        const uint32_t per = keymap_reply_->keysyms_per_keycode;
        if (per == 0u) return 0u;
        const uint32_t index = static_cast<uint32_t>(keycode - keycode_min_) * per;
        if (index >= keysyms_len_) return 0u;
        return static_cast<uint32_t>(keymap_keysyms_[index]);
    }

    xcb_connection_t* connection_ = nullptr;
    xcb_screen_t*     screen_     = nullptr;
    xcb_window_t      window_     = 0;
    xcb_atom_t        delete_atom_ = 0;
    xcb_get_keyboard_mapping_reply_t* keymap_reply_ = nullptr;
    const xcb_keysym_t* keymap_keysyms_ = nullptr;
    uint32_t          keysyms_len_ = 0;
    uint8_t           keycode_min_ = 0;
    VkInstance        instance_   = VK_NULL_HANDLE;
    VkSurfaceKHR      surface_    = VK_NULL_HANDLE;
    uint32_t          width_      = 0;
    uint32_t          height_     = 0;
    bool              valid_      = false;
};

// ---------------------------------------------------------------------------
// HeadlessWindowSurface -- VK_EXT_headless_surface (modern loader / CI only).
//
// On the 1.2.131 loader here vkCreateHeadlessSurfaceEXT is NOT exported, so this
// backend is never selected by MakeSurface (LoaderExportsHeadlessSurface() ->
// false). It is included so the SAME binary runs on a modern box. The entry point
// is resolved at runtime via the loader (vkGetInstanceProcAddr) at CreateSurface.
// ---------------------------------------------------------------------------
class HeadlessWindowSurface final : public WindowSurface {
public:
    HeadlessWindowSurface(uint32_t width, uint32_t height)
        : width_(width), height_(height) {}

    ~HeadlessWindowSurface() override {
        if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
        }
    }

    WindowVkSurface CreateSurface(WindowVkInstance instance) override {
        if (surface_ != VK_NULL_HANDLE) {
            return reinterpret_cast<WindowVkSurface>(surface_);
        }
        instance_ = reinterpret_cast<VkInstance>(instance);
        auto create_fn = reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateHeadlessSurfaceEXT"));
        if (create_fn == nullptr) {
            return nullptr;  // not available on this loader -> caller skips
        }
        VkHeadlessSurfaceCreateInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
        if (create_fn(instance_, &info, nullptr, &surface_) != VK_SUCCESS) {
            surface_ = VK_NULL_HANDLE;
            return nullptr;
        }
        return reinterpret_cast<WindowVkSurface>(surface_);
    }

    void PollEvents(std::vector<WindowEvent>&) override {}  // headless: no events
    uint32_t Width() const override { return width_; }
    uint32_t Height() const override { return height_; }
    const char* BackendName() const override { return "headless"; }

    std::vector<std::string> RequiredInstanceExtensions() const override {
        return {"VK_KHR_surface", "VK_EXT_headless_surface"};
    }

private:
    VkInstance   instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_  = VK_NULL_HANDLE;
    uint32_t     width_    = 0;
    uint32_t     height_   = 0;
};

}  // namespace

std::unique_ptr<WindowSurface> MakeSurface(const std::string& title,
                                           uint32_t width, uint32_t height,
                                           SurfaceBackendKind* out_kind) {
    auto set_kind = [&](SurfaceBackendKind k) {
        if (out_kind != nullptr) *out_kind = k;
    };

    // 1. Headless (modern loader / CI) -- preferred when truly available.
    if (LoaderExportsHeadlessSurface() && InstanceExtAdvertised("VK_EXT_headless_surface")) {
        set_kind(SurfaceBackendKind::Headless);
        return std::make_unique<HeadlessWindowSurface>(width, height);
    }

    // 2. Xcb (this box, under Xvfb) -- requires a display + the xcb surface ext.
    if (DisplayAvailable() && InstanceExtAdvertised("VK_KHR_xcb_surface")) {
        auto xcb_surface = std::make_unique<XcbWindowSurface>(title, width, height);
        if (xcb_surface->valid()) {
            set_kind(SurfaceBackendKind::Xcb);
            return xcb_surface;
        }
    }

    set_kind(SurfaceBackendKind::None);
    return nullptr;
}

std::vector<std::string> PreferredSurfaceInstanceExtensions(SurfaceBackendKind* out_kind) {
    auto set_kind = [&](SurfaceBackendKind k) {
        if (out_kind != nullptr) *out_kind = k;
    };
    if (LoaderExportsHeadlessSurface() && InstanceExtAdvertised("VK_EXT_headless_surface")) {
        set_kind(SurfaceBackendKind::Headless);
        return {"VK_KHR_surface", "VK_EXT_headless_surface"};
    }
    if (DisplayAvailable() && InstanceExtAdvertised("VK_KHR_xcb_surface")) {
        set_kind(SurfaceBackendKind::Xcb);
        return {"VK_KHR_surface", "VK_KHR_xcb_surface"};
    }
    set_kind(SurfaceBackendKind::None);
    return {};
}

}  // namespace nuka::render::window
