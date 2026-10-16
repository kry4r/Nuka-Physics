// VK_EXT_headless_surface backend: a fixed-size surface with no window and no events,
// used where no video driver is available (CI, a display-less WSL session).

#include "render/window/window_surface.hpp"

#include <vulkan/vulkan.h>

#include <cstring>

namespace nuka::render::window {

namespace {

bool InstanceExtAdvertised(const char* name) {
    uint32_t count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS || count == 0u) {
        return false;
    }
    std::vector<VkExtensionProperties> exts(count);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, exts.data()) != VK_SUCCESS) return false;
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
}

class HeadlessWindowSurface final : public WindowSurface {
public:
    HeadlessWindowSurface(uint32_t width, uint32_t height) : width_(width), height_(height) {}

    ~HeadlessWindowSurface() override {
        if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
        }
    }

    WindowVkSurface CreateSurface(WindowVkInstance instance) override {
        if (surface_ != VK_NULL_HANDLE) return reinterpret_cast<WindowVkSurface>(surface_);
        instance_ = reinterpret_cast<VkInstance>(instance);
        auto create_fn = reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateHeadlessSurfaceEXT"));
        if (create_fn == nullptr) return nullptr;
        VkHeadlessSurfaceCreateInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
        if (create_fn(instance_, &info, nullptr, &surface_) != VK_SUCCESS) {
            surface_ = VK_NULL_HANDLE;
            return nullptr;
        }
        return reinterpret_cast<WindowVkSurface>(surface_);
    }

    void PollEvents(std::vector<WindowEvent>&) override {}
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

std::unique_ptr<WindowSurface> MakeHeadlessSurface(uint32_t width, uint32_t height) {
    if (!InstanceExtAdvertised("VK_EXT_headless_surface")) return nullptr;
    return std::make_unique<HeadlessWindowSurface>(width, height);
}

}  // namespace nuka::render::window
